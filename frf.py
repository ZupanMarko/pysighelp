"""
Frequency Response Function (FRF) Analysis Utilities

This module contains functions for calculating frequency response functions,
coherence, and related transfer function analysis.
"""

import numpy as np
import scipy.signal as signal

def calculate_fft(signal, fs):
    """
    Calculate the FFT of a signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal to be transformed.
    fs : float
        Sampling frequency of the signal.
        
    Returns
    -------
    freq : ndarray
        Frequency array.
    fft_result : ndarray
        FFT of the input signal.
    """
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft_result = np.fft.rfft(signal)
    
    return freq, fft_result


def calculate_frf_fft(response, excitation, fs):
    """
    Calculate Frequency Response Function using FFT method (H1, H2 estimators) and coherence.
    
    Parameters
    ----------
    response : array_like
        Response signal (output).
    excitation : array_like
        Excitation signal (input).
    fs : float
        Sampling frequency.
        
    Returns
    -------
    freq : ndarray
        Frequency array.
    H1 : ndarray
        H1 frequency response function estimate.
    H2 : ndarray
        H2 frequency response function estimate.
    Coh : ndarray
        Coherence function.
    """
    N = len(response)
    freq = np.fft.rfftfreq(N, d=1/fs)
    response_fft = np.fft.rfft(response)
    excitation_fft = np.fft.rfft(excitation)

    # Power spectral densities
    Pff = excitation_fft * np.conj(excitation_fft)
    Paa = response_fft * np.conj(response_fft)
    Paf = response_fft * np.conj(excitation_fft)
    Pfa = excitation_fft * np.conj(response_fft)

    # H1 and H2 estimators
    H1 = Paf / Pff
    H2 = Paa / Pfa

    # Coherence
    Coh = np.abs(Paf)**2 / (Pff * Paa)

    return freq, H1, H2, Coh

def calculate_frf_welch(response, excitation, fs, nperseg=1024, noverlap=512):
    """
    Calculate Frequency Response Function using Welch's method (H1 and H2 estimators).
    
    Parameters
    ----------
    response : array_like
        Response signal (output).
    excitation : array_like
        Excitation signal (input).
    fs : float
        Sampling frequency.
    nperseg : int, optional
        Length of each segment. Default is 1024.
    noverlap : int, optional
        Number of points to overlap between segments. Default is 512.
        
    Returns
    -------
    f : ndarray
        Frequency array.
    H1 : ndarray
        H1 frequency response function estimate.
    H2 : ndarray
        H2 frequency response function estimate.
    Coh : ndarray
        Coherence function.
    """
    # Cross power spectral densities
    f, Paf = signal.csd(response, excitation, fs, nperseg=nperseg, noverlap=noverlap)
    _, Pfa = signal.csd(excitation, response, fs, nperseg=nperseg, noverlap=noverlap)
    # Power spectral densities
    _, Pff = signal.welch(excitation, fs, nperseg=nperseg, noverlap=noverlap)
    _, Paa = signal.welch(response, fs, nperseg=nperseg, noverlap=noverlap)
    # H1 and H2 estimators
    H1 = Paf / Pff
    H2 = Paa / Pfa
    # Coherence
    _, Coh = signal.coherence(response, excitation, fs, nperseg=nperseg, noverlap=noverlap)
    return f, H1, H2, Coh

def calculate_frf_spectrogram(response, excitation, fs, nperseg=1024, noverlap=512):
    """
    Calculate time-varying frequency response function using STFT.

    Parameters
    ----------
    response : array_like
        Response signal (output).
    excitation : array_like
        Excitation signal (input).
    fs : float
        Sampling frequency.
    nperseg : int, optional
        Length of each segment. Default is 8*1024.
    noverlap : int, optional
        Number of points to overlap between segments. Default is 8*512.

    Returns
    -------
    f : ndarray
        Frequency array.
    t : ndarray
        Time array.
    H : ndarray
        Time-varying frequency response function.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> f, t, H = calculate_frf_spectrogram(response, excitation, fs)
    >>> plt.pcolormesh(t, f, 20 * np.log10(np.abs(H)), shading='gouraud')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.title('FRF Spectrogram (Magnitude, dB)')
    >>> plt.colorbar(label='Magnitude [dB]')
    >>> plt.show()
    """
    f_response, t_response, Zxx_response = signal.stft(response, fs, nperseg=nperseg, noverlap=noverlap)
    f_excitation, t_excitation, Zxx_excitation = signal.stft(excitation, fs, nperseg=nperseg, noverlap=noverlap)

    # Calculate the time-varying FRF
    H = Zxx_response / Zxx_excitation

    return f_response, t_response, H

