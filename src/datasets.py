"""
datasets.py
-----------
Synthetic dataset generators and a catalog of bundled datasets.

Provides functions to generate spike trains with various statistical
properties (Poisson, bursty, rhythmic, correlated) and a registry
of pre-generated CSV files shipped with the project.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BUNDLED_DATASETS = {
    "sample_spikes": {
        "file": "sample_spikes.csv",
        "description": "5 neurons, 10 s, independent Poisson processes",
        "n_neurons": 5,
        "duration": 10,
    },
    "burst_neurons": {
        "file": "burst_neurons.csv",
        "description": "8 neurons with burst-firing patterns",
        "n_neurons": 8,
        "duration": 10,
    },
    "rhythmic_spikes": {
        "file": "rhythmic_spikes.csv",
        "description": "6 neurons with ~8 Hz oscillatory modulation",
        "n_neurons": 6,
        "duration": 10,
    },
    "large_population": {
        "file": "large_population.csv",
        "description": "50 neurons, 60 s of Poisson activity",
        "n_neurons": 50,
        "duration": 60,
    },
    "correlated_pairs": {
        "file": "correlated_pairs.csv",
        "description": "4 correlated neuron pairs (8 neurons total)",
        "n_neurons": 8,
        "duration": 10,
    },
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_poisson_spikes(n_neurons, duration, rates=10.0, seed=None):
    """
    Generate synthetic spike trains from independent Poisson processes.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate.
    duration : float
        Recording duration in seconds.
    rates : float or array-like
        Firing rate(s) in Hz.  A single float applies the same rate to
        every neuron; a sequence of length *n_neurons* sets per-neuron
        rates.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from 1-based neuron ID to sorted spike-time arrays.
    """
    rng = np.random.default_rng(seed)

    if np.isscalar(rates):
        rates = [float(rates)] * n_neurons
    else:
        rates = list(rates)
        if len(rates) != n_neurons:
            raise ValueError(
                f"Length of rates ({len(rates)}) must equal "
                f"n_neurons ({n_neurons})."
            )

    spikes = {}
    for i in range(n_neurons):
        rate = rates[i]
        if rate <= 0:
            spikes[i + 1] = np.array([], dtype=float)
            continue

        # Expected number of spikes, with some headroom
        n_expected = int(rate * duration * 1.5) + 10
        isis = rng.exponential(1.0 / rate, size=n_expected)
        times = np.cumsum(isis)
        times = times[times < duration]
        spikes[i + 1] = np.round(times, 4)

    return spikes


def generate_burst_spikes(
    n_neurons,
    duration,
    burst_rate=2.0,
    burst_size=5,
    intra_burst_freq=100.0,
    seed=None,
):
    """
    Generate spike trains with burst-firing patterns.

    Each neuron fires bursts at *burst_rate* Hz.  Each burst contains
    *burst_size* spikes with intra-burst intervals drawn from an
    exponential distribution centred at ``1 / intra_burst_freq``.

    Parameters
    ----------
    n_neurons : int
    duration : float
    burst_rate : float
        Average number of bursts per second.
    burst_size : int
        Number of spikes per burst.
    intra_burst_freq : float
        Mean intra-burst spike frequency in Hz.
    seed : int, optional

    Returns
    -------
    dict[int, np.ndarray]
    """
    rng = np.random.default_rng(seed)
    spikes = {}

    for i in range(n_neurons):
        # Generate burst onset times via Poisson process
        n_bursts_expected = int(burst_rate * duration * 1.5) + 10
        burst_isis = rng.exponential(1.0 / burst_rate, size=n_bursts_expected)
        burst_onsets = np.cumsum(burst_isis)
        burst_onsets = burst_onsets[burst_onsets < duration]

        all_times = []
        for onset in burst_onsets:
            intra_isis = rng.exponential(
                1.0 / intra_burst_freq, size=burst_size
            )
            burst_times = onset + np.cumsum(intra_isis)
            all_times.extend(burst_times[burst_times < duration])

        times = np.sort(np.round(np.array(all_times), 4))
        spikes[i + 1] = times

    return spikes


def generate_rhythmic_spikes(
    n_neurons,
    duration,
    freq=8.0,
    jitter=0.01,
    base_rate=5.0,
    seed=None,
):
    """
    Generate spike trains with oscillatory / rhythmic modulation.

    The instantaneous firing rate is modulated by a sinusoidal function
    at the given frequency, producing theta-like oscillatory activity.

    Parameters
    ----------
    n_neurons : int
    duration : float
    freq : float
        Modulation frequency in Hz (default 8 Hz = theta band).
    jitter : float
        Standard deviation of Gaussian jitter added to spike times (s).
    base_rate : float
        Baseline firing rate in Hz.
    seed : int, optional

    Returns
    -------
    dict[int, np.ndarray]
    """
    rng = np.random.default_rng(seed)
    spikes = {}
    dt = 0.001  # 1 ms resolution

    t = np.arange(0, duration, dt)
    # Sinusoidal rate modulation (always non-negative)
    rate_envelope = base_rate * (1.0 + np.sin(2.0 * np.pi * freq * t))

    for i in range(n_neurons):
        # Phase offset per neuron
        phase = rng.uniform(0, 2 * np.pi)
        neuron_rate = base_rate * (
            1.0 + np.sin(2.0 * np.pi * freq * t + phase)
        )

        # Inhomogeneous Poisson: spike probability per bin
        prob = neuron_rate * dt
        spike_mask = rng.random(len(t)) < prob
        times = t[spike_mask]

        # Add jitter
        if jitter > 0 and len(times) > 0:
            times = times + rng.normal(0, jitter, size=len(times))
            times = np.clip(times, 0, duration)

        spikes[i + 1] = np.sort(np.round(times, 4))

    return spikes


def generate_correlated_spikes(
    n_neurons,
    duration,
    correlation=0.3,
    base_rate=10.0,
    seed=None,
):
    """
    Generate correlated spike trains using a shared latent process.

    Neurons are grouped in pairs.  Each pair shares a fraction
    (*correlation*) of spikes from a common latent Poisson process,
    with the remainder generated independently.

    Parameters
    ----------
    n_neurons : int
        Must be even (neurons are grouped in pairs).
    duration : float
    correlation : float
        Fraction of spikes from the shared process (0 to 1).
    base_rate : float
        Average firing rate per neuron in Hz.
    seed : int, optional

    Returns
    -------
    dict[int, np.ndarray]
    """
    if n_neurons % 2 != 0:
        raise ValueError("n_neurons must be even for correlated pairs.")

    rng = np.random.default_rng(seed)
    spikes = {}

    shared_rate = base_rate * correlation
    private_rate = base_rate * (1.0 - correlation)

    n_pairs = n_neurons // 2
    for pair_idx in range(n_pairs):
        # Shared spike train
        shared = _poisson_times(shared_rate, duration, rng)

        for offset in range(2):
            neuron_id = pair_idx * 2 + offset + 1
            private = _poisson_times(private_rate, duration, rng)
            combined = np.sort(np.concatenate([shared, private]))
            spikes[neuron_id] = np.round(combined, 4)

    return spikes


def _poisson_times(rate, duration, rng):
    """Helper: generate Poisson spike times."""
    if rate <= 0:
        return np.array([], dtype=float)
    n_expected = int(rate * duration * 1.5) + 10
    isis = rng.exponential(1.0 / rate, size=n_expected)
    times = np.cumsum(isis)
    return times[times < duration]


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def list_datasets():
    """
    Return metadata for all bundled datasets.

    Returns
    -------
    dict
        Keys are dataset names; values are metadata dicts with fields
        ``file``, ``description``, ``n_neurons``, ``duration``.
    """
    return dict(BUNDLED_DATASETS)


def load_dataset(name):
    """
    Load a bundled dataset by name.

    Parameters
    ----------
    name : str
        One of the keys in :func:`list_datasets`.

    Returns
    -------
    dict[int, np.ndarray]

    Raises
    ------
    KeyError
        If *name* is not a recognised dataset.
    FileNotFoundError
        If the data file is missing.
    """
    if name not in BUNDLED_DATASETS:
        available = ", ".join(sorted(BUNDLED_DATASETS))
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available}"
        )

    from . import loader

    filepath = DATA_DIR / BUNDLED_DATASETS[name]["file"]
    return loader.load_spike_csv(filepath)


def save_spikes_csv(spikes_by_neuron, filepath):
    """
    Save a spike-train dictionary to CSV.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
        Mapping from neuron ID to spike-time arrays.
    filepath : str or Path
        Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write("neuron_id,spike_time\n")
        for neuron_id in sorted(spikes_by_neuron.keys()):
            for t in spikes_by_neuron[neuron_id]:
                f.write(f"{neuron_id},{t:.4f}\n")
