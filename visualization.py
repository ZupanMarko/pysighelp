import matplotlib.pyplot as plt
import numpy as np

def plot_time_signals(signals, time, y_range = []):
    """
    Plot multiple time-domain signals on subplots.
    
    Parameters
    ----------
    signals : list of np.ndarray
        List of signals to be plotted.
    time : np.ndarray
        Time vector corresponding to the signals.
    y_range : tuple, optional
        Y-axis limits as (min, max). Default is empty, which means auto-scaling.
    
    Returns
    -------
    None
    """
    N = len(signals)
    fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)

    for i, signal in enumerate(signals):
        axes[i].plot(time, signal)
        axes[i].set_title(f'Signal {i+1}')
        axes[i].set_ylabel('Amplitude')
        if y_range:
            axes[i].set_ylim(y_range)
        axes[i].grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_ni_time_signals(results, y_range=[]):
    """
    Plot time-domain signals from NI results.

    Parameters
    ----------
    results : dict
        Dictionary containing 't' and signal keys.
    y_range : tuple, optional
        Y-axis limits as (min, max). Default is empty, which means auto-scaling.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    t = results.get('t', None)
    if t is None:
        raise ValueError("Time vector 't' not found in results.")

    signal_names = []
    signals = []

    for key, value in results.items():
        if key == 't':
            continue
        if isinstance(value, (np.ndarray, list)):
            signals.append(np.array(value))
            signal_names.append(key)
        else:
            raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

    N = len(signals)
    fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)

    # If only one signal, axes is not a list
    if N == 1:
        axes = [axes]

    for i, signal in enumerate(signals):
        axes[i].plot(t, signal, alpha=0.7)
        axes[i].set_title(f'Signal: {signal_names[i]}')
        axes[i].set_ylabel('Amplitude')
        if y_range:
            axes[i].set_ylim(y_range)
        axes[i].grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


