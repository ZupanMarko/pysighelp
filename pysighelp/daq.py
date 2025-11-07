import numpy as np
from pysighelp.filter import low_pass_filter, high_pass_filter
from scipy.signal import iirnotch, filtfilt

class WheatstoneBridge:
    """
    Class to model a Wheatstone Bridge circuit for strain gauge measurements.

    ...

    Attributes
    ----------
    VA: np.ndarray
        Voltage at node A of the Wheatstone Bridge
    VB: np.ndarray
        Voltage at node B of the Wheatstone Bridge
    Vout: np.ndarray
        Output voltage of the Wheatstone Bridge
    Rfixed: float
        Resistance of static resistors in Wheatstone Bridge PCB (Ohms)
    Vexec: float
        Excitation voltage applied to the Wheatstone Bridge (Volts)
    mode: str
        Configuration mode of the Wheatstone Bridge ('quarter', 'half')     
    fs: float
        Sampling frequency (Hz)
    """

    def __init__(self,  VA: np.ndarray, VB: np.ndarray, Vout: np.ndarray, Rfixed=15000, Vexec=12.0, mode='quarter', fs=51200.0):
        self.VA = VA
        self.VB = VB
        self.Vout = Vout
        self.Rfixed = Rfixed
        self.Vexec = Vexec
        self.mode = mode.lower()
        self.fs = fs
        if self.mode not in ['quarter', 'half']:
            raise ValueError("Mode must be 'quarter' or 'half'")
        

    def filter(self, lowcut = None, highcut = None, order=5):
        """
        Filters the Wheatstone Bridge voltages using high-pass and/or low-pass filters.

        Parameters
        ----------
        lowcut : float or None
            Lower cutoff frequency for high-pass filter (Hz). If None, no high-pass filter is applied.
        highcut : float or None
            Upper cutoff frequency for low-pass filter (Hz). If None, no low-pass filter is applied.
        order : int
            Order of the filter (default is 5).
        """

        if lowcut is not None:
            self.VA = high_pass_filter(self.VA, lowcut, self.fs, order=order)
            self.VB = high_pass_filter(self.VB, lowcut, self.fs, order=order)
            self.Vout = high_pass_filter(self.Vout, lowcut, self.fs, order=order)

        if highcut is not None:
            self.VA = low_pass_filter(self.VA, highcut, self.fs, order=order)
            self.VB = low_pass_filter(self.VB, highcut, self.fs, order=order)
            self.Vout = low_pass_filter(self.Vout, highcut, self.fs, order=order)



    def compute_gain(self, Vdiff, Vout, prefilter_notch=True, f0=50.0, q=30.0, do_robust=True, trim_mad=4.0):
        """
        Estimate linear gain between Vdiff and Vout using least-squares regression,
        optionally prefiltering with a notch and doing a robust re-fit using MAD trimming.

        Returns
        -------
        gain : float
            Estimated slope (Vout = gain * Vdiff + intercept)
        intercept : float
            Estimated intercept (offset)
        """
        Vd = Vdiff.copy()
        Vo = Vout.copy()

        if prefilter_notch:
            nyq = 0.5 * self.fs
            w0 = f0 / nyq
            if w0 <= 0 or w0 >= 1:
                raise ValueError("Check fs / f0 values: normalized f0 must be in (0,1).")
            b, a = iirnotch(w0, q)
            if len(Vd) > 3 * max(len(a), len(b)):
                Vd = filtfilt(b, a, Vd)
                Vo = filtfilt(b, a, Vo)
            else:
                Vd = Vd - np.mean(Vd)
                Vo = Vo - np.mean(Vo)

        Vd = Vd - np.mean(Vd)
        Vo = Vo - np.mean(Vo)

        # First pass: least-squares slope + intercept
        A = np.vstack([Vd, np.ones_like(Vd)]).T
        slope, intercept = np.linalg.lstsq(A, Vo, rcond=None)[0]

        if do_robust:
            # residuals and MAD-based trimming
            resid = Vo - (slope * Vd + intercept)
            mad = np.median(np.abs(resid - np.median(resid)))
            if mad == 0:
                # fallback to std if MAD is zero (rare)
                sigma = np.std(resid)
            else:
                sigma = 1.4826 * mad  # consistent estimator of sigma from MAD
            mask = np.abs(resid) <= (trim_mad * sigma)
            if np.sum(mask) < 0.5 * len(Vd):
                pass
            else:
                A2 = np.vstack([Vd[mask], np.ones(np.sum(mask))]).T
                slope, intercept = np.linalg.lstsq(A2, Vo[mask], rcond=None)[0]
        

        return float(slope), float(intercept)


    def compute(self, prefilter_notch=False, notch_f0=50.0, notch_q=30.0, robust_gain=True, gain_postcorrection=True):
        """
        Compute time-dependent and static resistances using a robust gain estimate.

        Optional args control pre-filtering for gain estimation.
        """
        N = len(self.VA)
        t = np.arange(N) / self.fs

        Vdiff = self.VA - self.VB

        if self.mode == 'quarter':
            self.Rb = np.mean((self.VA) / (self.Vexec - self.VA) * self.Rfixed)
        elif self.mode == 'half':
            self.Rb = 2 * np.mean((self.VA) / (self.Vexec - self.VA) * self.Rfixed)

        # Non-amplified resistance (from differential)
        Rna = self.Rb * (1 + 4 * Vdiff / self.Vexec)

        slope, intercept = self.compute_gain(Vdiff, self.Vout,
                                            prefilter_notch=prefilter_notch,
                                            f0=notch_f0, q=notch_q,
                                            do_robust=robust_gain)
        self.gain = slope if slope != 0 else 1.0
        self.gain_intercept = intercept

        R = self.Rb * (1 + 4 * (self.Vout - self.gain_intercept) / (self.gain * self.Vexec))

        #Gain postcorrection
        if gain_postcorrection: 
            correction_factor = np.mean(Rna) / np.mean(R)
            R *= correction_factor

        R0 = float(np.mean(Rna))

        self.t = t
        self.R = R
        self.Rna = Rna
        self.R0 = R0
        return t, R, Rna, R0


    def plot(self, **kwargs):
        """
        Plot the Wheatstone Bridge resistance.
        """
        import matplotlib.pyplot as plt

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.8
        if 'color' not in kwargs:
            kwargs['color'] = 'black'

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.R, **kwargs)
        plt.title("Wheatstone Bridge Resistance")
        plt.xlabel("Time [s]")
        plt.ylabel("Resistance [Ohms]")
        plt.grid()
        plt.show()

