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


def load_nwb(filepath):
    """
    Load spike-train data from an NWB (Neurodata Without Borders) file.

    Extracts spike times from the ``Units`` table, which is the
    standard NWB container for sorted spike data.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``.nwb`` file.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from unit / neuron id (1-based) to sorted spike times.

    Raises
    ------
    ImportError
        If *pynwb* is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the NWB file contains no ``Units`` table.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        raise ImportError(
            "pynwb is required to load NWB files.\n"
            "Install with:  pip install pynwb"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NWB file not found: {filepath}")

    spikes_by_neuron = {}

    with NWBHDF5IO(str(filepath), "r") as io:
        nwbfile = io.read()

        if nwbfile.units is None:
            raise ValueError(
                f"No Units table found in {filepath}. "
                "The file may contain raw data but no sorted spikes."
            )

        for idx in range(len(nwbfile.units)):
            spike_times = nwbfile.units["spike_times"][idx]
            neuron_id = idx + 1  # 1-based
            spikes_by_neuron[neuron_id] = np.sort(
                np.asarray(spike_times, dtype=float)
            )

    return spikes_by_neuron


def load_auto(filepath):
    """
    Auto-detect file format and load spike data.

    Dispatches to :func:`load_spike_csv` for ``.csv`` files and
    :func:`load_nwb` for ``.nwb`` files.

    Parameters
    ----------
    filepath : str or Path
        Path to a spike data file (``.csv`` or ``.nwb``).

    Returns
    -------
    dict[int, np.ndarray]

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return load_spike_csv(filepath)
    elif suffix == ".nwb":
        return load_nwb(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Supported formats: .csv, .nwb"
        )
