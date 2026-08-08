"""
visualization.py
-----------------
Matplotlib-based plotting helpers for spike train data:
raster plots, firing rate curves, ISI histograms, and
multi-neuron comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import analysis


def plot_raster(spikes_by_neuron, ax=None, title="Spike Raster Plot"):
    """
    Plot a raster of spikes: one row per neuron, one tick per spike.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    neuron_ids = sorted(spikes_by_neuron.keys())
    for row, neuron_id in enumerate(neuron_ids):
        times = spikes_by_neuron[neuron_id]
        ax.vlines(times, row + 0.6, row + 1.4, color="black", linewidth=0.8)

    ax.set_yticks(range(1, len(neuron_ids) + 1))
    ax.set_yticklabels([f"Neuron {n}" for n in neuron_ids])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(0.5, len(neuron_ids) + 0.5)
    return ax


def plot_firing_rate_over_time(spikes_by_neuron, bin_size=0.5, ax=None,
                                title="Population Firing Rate"):
    """
    Plot the population-averaged firing rate (Hz) over time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    bin_centers, rate_hz = analysis.population_rate(
        spikes_by_neuron, bin_size=bin_size
    )
    ax.plot(bin_centers, rate_hz, color="steelblue", linewidth=1.5)
    ax.fill_between(bin_centers, rate_hz, alpha=0.2, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    return ax


def plot_spike_histogram(spike_times, bin_size=0.5, ax=None,
                          title="Spike Count Histogram"):
    """
    Plot a histogram of spike counts per time bin for a single neuron.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    bin_edges, counts = analysis.binned_spike_counts(spike_times, bin_size)
    ax.bar(bin_edges[:-1], counts, width=bin_size * 0.9,
           align="edge", color="darkorange", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike count")
    ax.set_title(title)
    return ax


def plot_isi_histogram(spike_times, bins=30, ax=None,
                        title="Inter-Spike Interval Distribution"):
    """
    Plot a histogram of inter-spike intervals (ISIs) for a single neuron.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    isis = analysis.inter_spike_intervals(spike_times)
    ax.hist(isis, bins=bins, color="mediumseagreen", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("ISI (s)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def plot_firing_rate_comparison(spikes_by_neuron, duration=None, ax=None,
                                 title="Firing Rate by Neuron"):
    """
    Bar chart comparing average firing rate across neurons.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    rates = analysis.firing_rates_all(spikes_by_neuron, duration=duration)
    neuron_ids = sorted(rates.keys())
    values = [rates[n] for n in neuron_ids]

    ax.bar([str(n) for n in neuron_ids], values, color="slateblue", edgecolor="black")
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    return ax
