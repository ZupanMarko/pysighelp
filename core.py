"""
General numerical methods for scientific computing.
Extracted from bearing analysis scripts with nonlinear solving and interpolation.
"""
import numpy as np
from scipy.optimize import fsolve, newton, root_scalar
from scipy.interpolate import interp1d, CubicSpline
import sympy as sp
from typing import Callable, Optional, Tuple, Any, Union

def differentiate_function(func: Callable, x: Union[float, np.ndarray], 
                         h: float = 1e-8) -> Union[float, np.ndarray]:
    """
    Numerical differentiation using central difference.
    
    Args:
        func: Function to differentiate
        x: Point(s) to evaluate derivative
        h: Step size
        
    Returns:
        Derivative value(s)
    """
    return (func(x + h) - func(x - h)) / (2 * h)

def integrate_function(func: Callable, a: float, b: float, 
                      method: str = 'quad') -> float:
    """
    Numerical integration.
    
    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        method: Integration method
        
    Returns:
        Integral value
    """
    from scipy.integrate import quad, trapz
    
    if method == 'quad':
        result, _ = quad(func, a, b)
        return result
    elif method == 'trapz':
        x = np.linspace(a, b, 1000)
        y = func(x)
        return trapz(y, x)
    else:
        raise ValueError(f"Unknown integration method: {method}")

def freq_domain_differentiate(signal: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Differentiate a signal in the frequency domain (1/s).
    Args:
        signal: Frequency domain signal (complex or real)
        freqs: Frequencies (Hz)
    Returns:
        Differentiated signal in frequency domain
    """
    omega = 2 * np.pi * freqs
    return 1j * omega * signal

def freq_domain_integrate(signal: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Integrate a signal in the frequency domain (*s).
    Args:
        signal: Frequency domain signal (complex or real)
        freqs: Frequencies (Hz)
    Returns:
        Integrated signal in frequency domain
    """
    omega = 2 * np.pi * freqs
    # Avoid division by zero at DC
    omega_safe = np.where(omega == 0, np.finfo(float).eps, omega)
    return signal / (1j * omega_safe)

def time_domain_differentiate(signal: np.ndarray, dt: float) -> np.ndarray:
    """
    Differentiate a real signal in the time domain using frequency domain method.
    Args:
        signal: Time domain signal (real)
        dt: Sampling interval (seconds)
    Returns:
        Differentiated signal in time domain (real)
    """
    n = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, dt)
    signal_fft = np.fft.rfft(signal)
    diff_fft = freq_domain_differentiate(signal_fft, freqs)
    diff_time = np.fft.irfft(diff_fft, n=n)
    return diff_time

def time_domain_integrate(signal: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate a real signal in the time domain using frequency domain method.
    Args:
        signal: Time domain signal (real)
        dt: Sampling interval (seconds)
    Returns:
        Integrated signal in time domain (real)
    """
    n = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, dt)
    signal_fft = np.fft.rfft(signal)
    integ_fft = freq_domain_integrate(signal_fft, freqs)
    integ_time = np.fft.irfft(integ_fft, n=n)
    return integ_time
