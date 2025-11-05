"""
Filtering Utilities

This module contains various digital filtering functions for signal processing,
including low-pass, high-pass, band-pass, and FIR filters.
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, firwin


def low_pass_filter(signal, cutoff_freq, sample_rate, order=6):
    """
    Apply a low-pass filter to a given signal to filter out higher frequencies.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency of the low-pass filter.
    sample_rate : float
        The sample rate of the signal.
    order : int, optional
        The order of the filter (default is 6).

    Returns
    -------
    filtered_signal : numpy.ndarray
        The filtered signal.
    """
    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Design a Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the signal using filtfilt to ensure zero phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def high_pass_filter(signal, cutoff_freq, sample_rate, order=6):
    """
    Apply a high-pass filter to a given signal to filter out lower frequencies.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency of the high-pass filter.
    sample_rate : float
        The sample rate of the signal.
    order : int, optional
        The order of the filter (default is 6).

    Returns
    -------
    filtered_signal : numpy.ndarray
        The filtered signal.
    """
    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Design a Butterworth high-pass filter
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # Apply the filter to the signal using filtfilt to ensure zero phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal





def band_pass_filter_FIR(signal_data, lowcut, highcut, sample_rate, order=101):
    """
    Apply an FIR bandpass filter to the input signal.
    
    Parameters
    ----------
    signal_data : numpy.ndarray
        The input signal (1D numpy array).
    lowcut : float
        The lower cutoff frequency in Hz.
    highcut : float
        The upper cutoff frequency in Hz.
    sample_rate : float
        The sampling rate of the signal in Hz.
    order : int, optional
        The number of taps (filter order, default is 101).
    
    Returns
    -------
    filtered_signal : numpy.ndarray
        The filtered output signal.
    """
    # Normalize the cutoff frequencies with respect to the Nyquist frequency
    nyquist_rate = sample_rate / 2.0
    low = lowcut / nyquist_rate
    high = highcut / nyquist_rate
    
    # Design the FIR bandpass filter
    fir_coeff = firwin(order, [low, high], pass_zero=False)
    
    # Apply the filter to the signal using lfilter
    filtered_signal = lfilter(fir_coeff, 1.0, signal_data)
    
    return filtered_signal





def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to data.
    
    Parameters
    ----------
    data : array_like
        Input data to be filtered.
    lowcut : float
        Lower cutoff frequency.
    highcut : float
        Upper cutoff frequency.
    fs : float
        Sampling frequency.
    order : int, optional
        Filter order. Default is 5.
        
    Returns
    -------
    y : ndarray
        Filtered data.
    """
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    y = lfilter(b, a, data)
    return y


def notch_filter(signal, notch_freq, sample_rate, quality_factor=30):
    """
    Apply a notch filter to remove a specific frequency component.
    
    Parameters
    ----------
    signal : array_like
        Input signal.
    notch_freq : float
        Frequency to be removed (Hz).
    sample_rate : float
        Sampling frequency (Hz).
    quality_factor : float, optional
        Quality factor of the notch filter. Default is 30.
        
    Returns
    -------
    filtered_signal : ndarray
        Signal with the notch frequency removed.
    """
    from scipy.signal import iirnotch
    
    # Design the notch filter
    b, a = iirnotch(notch_freq, quality_factor, sample_rate)
    
    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def apply_window(signal, window_type='hann'):
    """
    Apply a window function to a signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal.
    window_type : str, optional
        Type of window function. Default is 'hann'.
        Options: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', etc.
        
    Returns
    -------
    windowed_signal : ndarray
        Signal with window function applied.
    """
    from scipy.signal import get_window
    
    window = get_window(window_type, len(signal))
    windowed_signal = signal * window
    
    return windowed_signal
