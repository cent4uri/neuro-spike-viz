"""
analysis.py
-----------
Analysis functions for neuronal spike trains.

Provides:
- firing-rate calculations
- inter-spike interval (ISI) analysis
- binned spike counts
- population firing rates
- spike-train summary statistics
"""

import numpy as np


def _validate_spike_times(spike_times):
    """
    Convert spike times to a clean 1D NumPy array.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.

    Returns
    -------
    np.ndarray
        Sorted spike times as floating-point values.

    Raises
    ------
    ValueError
        If spike times contain NaN or infinite values.
    """
    spike_times = np.asarray(spike_times, dtype=float).ravel()

    if not np.all(np.isfinite(spike_times)):
        raise ValueError("Spike times must contain only finite values.")

    return np.sort(spike_times)


def _validate_duration(duration):
    """Validate recording duration."""
    if duration is None:
        return None

    duration = float(duration)

    if duration <= 0:
        raise ValueError("Duration must be greater than zero.")

    return duration


def _validate_bin_size(bin_size):
    """Validate bin size."""
    bin_size = float(bin_size)

    if bin_size <= 0:
        raise ValueError("Bin size must be greater than zero.")

    return bin_size


def spike_count(spike_times):
    """
    Count the number of spikes.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.

    Returns
    -------
    int
        Number of spikes.
    """
    spike_times = _validate_spike_times(spike_times)
    return len(spike_times)


def firing_rate(spike_times, duration=None):
    """
    Compute the average firing rate of a single neuron in Hz.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    duration : float, optional
        Total recording duration in seconds.

        If not provided, the duration is estimated from the first
        and last spike.

    Returns
    -------
    float
        Average firing rate in spikes per second (Hz).
    """
    spike_times = _validate_spike_times(spike_times)

    if len(spike_times) == 0:
        return 0.0

    duration = _validate_duration(duration)

    if duration is None:
        duration = spike_times[-1] - spike_times[0]

        if duration <= 0:
            return 0.0

    return len(spike_times) / duration


def firing_rates_all(spikes_by_neuron, duration=None):
    """
    Compute firing rates for all neurons.

    Parameters
    ----------
    spikes_by_neuron : dict
        Mapping of neuron ID to spike-time arrays.
    duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    dict
        Mapping of neuron ID to firing rate in Hz.
    """
    return {
        neuron_id: firing_rate(times, duration=duration)
        for neuron_id, times in spikes_by_neuron.items()
    }


def inter_spike_intervals(spike_times):
    """
    Compute inter-spike intervals (ISI).

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.

    Returns
    -------
    np.ndarray
        Consecutive spike intervals in seconds.
    """
    spike_times = _validate_spike_times(spike_times)

    if len(spike_times) < 2:
        return np.array([], dtype=float)

    return np.diff(spike_times)


def mean_isi(spike_times):
    """
    Calculate the mean inter-spike interval.

    Returns
    -------
    float
        Mean ISI in seconds.
        Returns 0.0 if fewer than two spikes exist.
    """
    isi = inter_spike_intervals(spike_times)

    if len(isi) == 0:
        return 0.0

    return float(np.mean(isi))


def median_isi(spike_times):
    """
    Calculate the median inter-spike interval.

    Returns
    -------
    float
        Median ISI in seconds.
        Returns 0.0 if fewer than two spikes exist.
    """
    isi = inter_spike_intervals(spike_times)

    if len(isi) == 0:
        return 0.0

    return float(np.median(isi))


def isi_coefficient_of_variation(spike_times):
    """
    Calculate the coefficient of variation (CV) of ISIs.

    CV = standard deviation of ISI / mean ISI.

    Returns
    -------
    float
        ISI coefficient of variation.
        Returns 0.0 if fewer than two spikes exist or mean ISI is zero.
    """
    isi = inter_spike_intervals(spike_times)

    if len(isi) == 0:
        return 0.0

    mean = np.mean(isi)

    if mean == 0:
        return 0.0

    return float(np.std(isi) / mean)


def binned_spike_counts(spike_times, bin_size=0.5, duration=None):
    """
    Count spikes within fixed-width time bins.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    bin_size : float, default=0.5
        Width of each time bin in seconds.
    duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        bin_edges:
            Boundaries of each time bin.

        counts:
            Number of spikes in each bin.
    """
    spike_times = _validate_spike_times(spike_times)
    bin_size = _validate_bin_size(bin_size)
    duration = _validate_duration(duration)

    if duration is None:
        duration = (
            float(spike_times[-1])
            if len(spike_times) > 0
            else bin_size
        )

    if duration <= 0:
        duration = bin_size

    bin_edges = np.arange(
        0,
        duration + bin_size,
        bin_size
    )

    # Ensure the final edge covers the requested duration.
    if bin_edges[-1] < duration:
        bin_edges = np.append(bin_edges, duration)

    counts, _ = np.histogram(
        spike_times,
        bins=bin_edges
    )

    return bin_edges, counts


def binned_firing_rate(spike_times, bin_size=0.5, duration=None):
    """
    Calculate firing rate within each time bin.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    bin_size : float, default=0.5
        Width of each bin in seconds.
    duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    bin_centers : np.ndarray
        Center of each time bin.

    rate_hz : np.ndarray
        Firing rate in Hz for each bin.
    """
    bin_edges, counts = binned_spike_counts(
        spike_times,
        bin_size=bin_size,
        duration=duration
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    rate_hz = counts / np.diff(bin_edges)

    return bin_centers, rate_hz


def population_rate(spikes_by_neuron, bin_size=0.5, duration=None):
    """
    Compute population-averaged firing rate over time.

    All neurons in ``spikes_by_neuron`` are included in the
    denominator, including neurons with zero spikes.

    Parameters
    ----------
    spikes_by_neuron : dict
        Mapping of neuron ID to spike-time arrays.
    bin_size : float, default=0.5
        Width of each time bin in seconds.
    duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    bin_centers : np.ndarray
        Center of each time bin.

    rate_hz : np.ndarray
        Average population firing rate in Hz.
    """
    bin_size = _validate_bin_size(bin_size)
    duration = _validate_duration(duration)

    if not spikes_by_neuron:
        duration = duration or bin_size

        bin_edges = np.arange(
            0,
            duration + bin_size,
            bin_size
        )

        if len(bin_edges) < 2:
            bin_edges = np.array([0, bin_size])

        bin_centers = (
            bin_edges[:-1] + bin_edges[1:]
        ) / 2

        return bin_centers, np.zeros(len(bin_centers))

    cleaned_spikes = {
        neuron_id: _validate_spike_times(times)
        for neuron_id, times in spikes_by_neuron.items()
    }

    non_empty = [
        times
        for times in cleaned_spikes.values()
        if len(times) > 0
    ]

    if duration is None:
        duration = (
            max(times[-1] for times in non_empty)
            if non_empty
            else bin_size
        )

    all_spikes = (
        np.concatenate(non_empty)
        if non_empty
        else np.array([], dtype=float)
    )

    bin_edges, counts = binned_spike_counts(
        all_spikes,
        bin_size=bin_size,
        duration=duration
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_neurons = len(cleaned_spikes)

    rate_hz = counts / (
        np.diff(bin_edges) * n_neurons
    )

    return bin_centers, rate_hz


def summarize_neuron(spike_times, duration=None):
    """
    Generate a summary of neuronal spike activity.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    dict
        Summary statistics for the neuron.
    """
    spike_times = _validate_spike_times(spike_times)

    isi = inter_spike_intervals(spike_times)

    return {
        "spike_count": spike_count(spike_times),
        "firing_rate_hz": firing_rate(
            spike_times,
            duration=duration
        ),
        "mean_isi_s": (
            float(np.mean(isi))
            if len(isi) > 0
            else 0.0
        ),
        "median_isi_s": (
            float(np.median(isi))
            if len(isi) > 0
            else 0.0
        ),
        "isi_cv": isi_coefficient_of_variation(
            spike_times
        ),
    }