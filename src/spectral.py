"""
spectral.py
-----------
Frequency-domain analysis of neuronal spike trains.

Converts spike times to binary time series and applies spectral
methods: power spectral density (PSD), spectrogram, coherence,
and band-power estimation.
"""

import numpy as np
from scipy import signal


def spike_train_to_binary(spike_times, bin_size=0.001, duration=None):
    """
    Convert spike times to a binary (0/1) time series.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    bin_size : float, default=0.001
        Bin width in seconds.  Determines the sampling rate
        (``fs = 1 / bin_size``).
    duration : float, optional
        Total duration in seconds.  Defaults to the time of the
        last spike.

    Returns
    -------
    binary_train : np.ndarray
        Binary array with 1 where a spike occurred.
    fs : float
        Sampling frequency in Hz (``1 / bin_size``).
    """
    spike_times = np.asarray(spike_times, dtype=float).ravel()
    spike_times = spike_times[np.isfinite(spike_times)]

    if len(spike_times) == 0:
        n_bins = max(1, int((duration or 1.0) / bin_size))
        return np.zeros(n_bins, dtype=float), 1.0 / bin_size

    if duration is None:
        duration = float(spike_times.max()) + bin_size

    fs = 1.0 / bin_size
    n_bins = int(np.ceil(duration / bin_size))
    bin_edges = np.arange(0, n_bins + 1) * bin_size

    counts, _ = np.histogram(spike_times, bins=bin_edges)
    binary_train = np.clip(counts, 0, 1).astype(float)

    return binary_train, fs


def power_spectral_density(binary_train, fs, nperseg=None):
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    binary_train : np.ndarray
        Binary spike-train time series.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each Welch segment.  Defaults to
        ``min(256, len(binary_train))``.

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz.
    psd : np.ndarray
        Power spectral density estimate.
    """
    if len(binary_train) == 0:
        return np.array([0.0]), np.array([0.0])

    if nperseg is None:
        nperseg = min(256, len(binary_train))

    freqs, psd = signal.welch(
        binary_train, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
    )
    return freqs, psd


def spike_spectrogram(binary_train, fs, nperseg=256, noverlap=None):
    """
    Compute a spectrogram of the binary spike train.

    Parameters
    ----------
    binary_train : np.ndarray
        Binary spike-train time series.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, default=256
        Segment length for the STFT.
    noverlap : int, optional
        Overlap between segments.  Defaults to ``nperseg // 2``.

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz.
    times : np.ndarray
        Time values in seconds.
    Sxx : np.ndarray
        Spectrogram power values (shape: frequencies × times).
    """
    if len(binary_train) == 0:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1))

    nperseg = min(nperseg, len(binary_train))

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = signal.spectrogram(
        binary_train, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    return freqs, times, Sxx


def dominant_frequency(binary_train, fs, nperseg=None):
    """
    Find the dominant (peak) frequency in the PSD.

    Parameters
    ----------
    binary_train : np.ndarray
    fs : float
    nperseg : int, optional

    Returns
    -------
    float
        Frequency with maximum power in Hz.  Returns 0.0 if the
        input is empty.
    """
    freqs, psd = power_spectral_density(binary_train, fs, nperseg)

    if len(psd) == 0 or np.all(psd == 0):
        return 0.0

    # Exclude DC component
    if len(freqs) > 1:
        idx = np.argmax(psd[1:]) + 1
    else:
        idx = 0

    return float(freqs[idx])


def band_power(binary_train, fs, low, high, nperseg=None):
    """
    Compute total power in a frequency band.

    Common neuroscience bands::

        delta :  0.5 –  4 Hz
        theta :  4   –  8 Hz
        alpha :  8   – 13 Hz
        beta  : 13   – 30 Hz
        gamma : 30   – 100 Hz

    Parameters
    ----------
    binary_train : np.ndarray
    fs : float
    low : float
        Lower frequency bound in Hz.
    high : float
        Upper frequency bound in Hz.
    nperseg : int, optional

    Returns
    -------
    float
        Integrated power in the band.
    """
    freqs, psd = power_spectral_density(binary_train, fs, nperseg)

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0

    return float(np.trapz(psd[mask], freqs[mask]))


def coherence(train_a, train_b, fs, nperseg=None):
    """
    Compute magnitude-squared coherence between two binary spike trains.

    Parameters
    ----------
    train_a : np.ndarray
        First binary spike-train time series.
    train_b : np.ndarray
        Second binary spike-train time series.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz.
    coh : np.ndarray
        Coherence values (0 to 1).
    """
    min_len = min(len(train_a), len(train_b))
    if min_len == 0:
        return np.array([0.0]), np.array([0.0])

    # Trim to equal length
    train_a = train_a[:min_len]
    train_b = train_b[:min_len]

    if nperseg is None:
        nperseg = min(256, min_len)

    freqs, coh = signal.coherence(
        train_a, train_b, fs=fs, nperseg=nperseg
    )
    return freqs, coh


# ---------------------------------------------------------------------------
# Convenience: standard band names
# ---------------------------------------------------------------------------

STANDARD_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def all_band_powers(binary_train, fs, bands=None, nperseg=None):
    """
    Compute power in each standard frequency band.

    Parameters
    ----------
    binary_train : np.ndarray
    fs : float
    bands : dict, optional
        Mapping of band name → (low, high).  Defaults to
        :data:`STANDARD_BANDS`.
    nperseg : int, optional

    Returns
    -------
    dict[str, float]
        Band name → integrated power.
    """
    if bands is None:
        bands = STANDARD_BANDS

    return {
        name: band_power(binary_train, fs, low, high, nperseg)
        for name, (low, high) in bands.items()
    }
