"""
analysis.py
-----------
Basic analysis functions for neuronal spike trains:
firing rates, inter-spike intervals, and binned spike counts.
"""

import numpy as np


def firing_rate(spike_times, duration=None):
    """
    Compute the average firing rate of a single neuron in Hz.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    duration : float, optional
        Total recording duration in seconds. If not given, uses the
        time span between the first and last spike.

    Returns
    -------
    float
        Average firing rate (spikes per second).
    """
    spike_times = np.asarray(spike_times)
    if len(spike_times) == 0:
        return 0.0

    if duration is None:
        duration = spike_times.max() - spike_times.min()
        if duration <= 0:
            return 0.0

    return len(spike_times) / duration


def firing_rates_all(spikes_by_neuron, duration=None):
    """
    Compute firing rate for every neuron in a spike dataset.

    Returns
    -------
    dict[int, float]
        Mapping from neuron id to firing rate (Hz).
    """
    return {
        neuron_id: firing_rate(times, duration=duration)
        for neuron_id, times in spikes_by_neuron.items()
    }


def inter_spike_intervals(spike_times):
    """
    Compute inter-spike intervals (ISIs) for a single neuron.

    Returns
    -------
    np.ndarray
        Array of ISIs in seconds (length = n_spikes - 1).
    """
    spike_times = np.sort(np.asarray(spike_times))
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times)


def binned_spike_counts(spike_times, bin_size=0.5, duration=None):
    """
    Bin spike times into fixed-width time windows and count spikes
    per bin. Useful for firing-rate-over-time plots.

    Parameters
    ----------
    spike_times : array-like
    bin_size : float
        Width of each time bin in seconds.
    duration : float, optional
        Total duration to bin over. Defaults to max spike time.

    Returns
    -------
    bin_edges : np.ndarray
    counts : np.ndarray
    """
    spike_times = np.asarray(spike_times)
    if duration is None:
        duration = spike_times.max() if len(spike_times) else bin_size

    bin_edges = np.arange(0, duration + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return bin_edges, counts


def population_rate(spikes_by_neuron, bin_size=0.5, duration=None):
    """
    Compute the population-averaged firing rate over time (Hz),
    combining spikes across all neurons.

    Returns
    -------
    bin_centers : np.ndarray
    rate_hz : np.ndarray
        Average firing rate per neuron, in Hz, within each bin.
    """
    all_spikes = np.concatenate(
        [times for times in spikes_by_neuron.values() if len(times) > 0]
    ) if spikes_by_neuron else np.array([])

    if duration is None:
        duration = all_spikes.max() if len(all_spikes) else bin_size

    n_neurons = max(len(spikes_by_neuron), 1)
    bin_edges, counts = binned_spike_counts(all_spikes, bin_size, duration)
    bin_centers = bin_edges[:-1] + bin_size / 2
    rate_hz = counts / (bin_size * n_neurons)
    return bin_centers, rate_hz
