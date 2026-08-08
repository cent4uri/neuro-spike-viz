"""
loader.py
---------
Utilities for loading neuronal spike train data from CSV files.

Expected CSV format:
    neuron_id,spike_time
    1,0.012
    1,0.083
    2,0.054
    ...

Where `spike_time` is in seconds.
"""

from pathlib import Path
import numpy as np


def load_spike_csv(filepath):
    """
    Load a spike-train CSV file into a dictionary mapping
    neuron_id -> sorted numpy array of spike times (seconds).

    Parameters
    ----------
    filepath : str or Path
        Path to a CSV file with columns `neuron_id,spike_time`.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from neuron id to an array of spike times.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Spike file not found: {filepath}")

    # Skip header row, load as (neuron_id, spike_time) pairs
    raw = np.genfromtxt(filepath, delimiter=",", skip_header=1)

    if raw.ndim == 1:
        # Only one row of data — reshape so indexing still works
        raw = raw.reshape(1, -1)

    spikes_by_neuron = {}
    for neuron_id, spike_time in raw:
        neuron_id = int(neuron_id)
        spikes_by_neuron.setdefault(neuron_id, []).append(spike_time)

    # Sort spike times and convert to numpy arrays
    for neuron_id in spikes_by_neuron:
        spikes_by_neuron[neuron_id] = np.sort(
            np.array(spikes_by_neuron[neuron_id])
        )

    return spikes_by_neuron


def neuron_ids(spikes_by_neuron):
    """Return a sorted list of neuron ids present in the dataset."""
    return sorted(spikes_by_neuron.keys())


def recording_duration(spikes_by_neuron):
    """
    Estimate the total duration (seconds) of the recording as the
    time of the last spike across all neurons.
    """
    last_spikes = [
        times.max() for times in spikes_by_neuron.values() if len(times) > 0
    ]
    return max(last_spikes) if last_spikes else 0.0
