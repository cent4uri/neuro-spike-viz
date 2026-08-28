"""
plotly_spectral.py
------------------
Interactive Plotly visualizations for spectral analysis results.
"""

import numpy as np
import plotly.graph_objects as go

from . import spectral


def plotly_psd(spike_times, bin_size=0.001, duration=None, nperseg=None,
               title="Power Spectral Density", log_y=True):
    """
    Interactive PSD plot.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    bin_size : float
        Bin width for binary conversion (determines Fs).
    duration : float, optional
    nperseg : int, optional
    title : str
    log_y : bool
        If True, use log-scaled y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    binary, fs = spectral.spike_train_to_binary(spike_times, bin_size, duration)
    freqs, psd = spectral.power_spectral_density(binary, fs, nperseg)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=psd,
        mode="lines",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="Freq: %{x:.1f} Hz<br>Power: %{y:.2e}<extra></extra>",
    ))

    yaxis_opts = dict(title="Power")
    if log_y:
        yaxis_opts["type"] = "log"

    fig.update_layout(
        title=title,
        xaxis=dict(title="Frequency (Hz)"),
        yaxis=yaxis_opts,
        template="plotly_white",
        width=900,
        height=400,
    )
    return fig


def plotly_spectrogram(spike_times, bin_size=0.001, duration=None,
                        nperseg=256, title="Spike Train Spectrogram"):
    """
    Interactive spectrogram heatmap.

    Parameters
    ----------
    spike_times : array-like
    bin_size : float
    duration : float, optional
    nperseg : int
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    binary, fs = spectral.spike_train_to_binary(spike_times, bin_size, duration)
    freqs, times, Sxx = spectral.spike_spectrogram(binary, fs, nperseg)

    # Use log scale for power (add small epsilon to avoid log(0))
    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Sxx_log,
        x=np.round(times, 3),
        y=np.round(freqs, 1),
        colorscale="Viridis",
        colorbar=dict(title="Power (dB)"),
        hovertemplate=(
            "Time: %{x:.3f} s<br>"
            "Freq: %{y:.1f} Hz<br>"
            "Power: %{z:.1f} dB<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Frequency (Hz)"),
        template="plotly_white",
        width=900,
        height=450,
    )
    return fig


def plotly_coherence(spike_times_a, spike_times_b, bin_size=0.001,
                      duration=None, nperseg=None,
                      title="Spike Train Coherence"):
    """
    Interactive coherence plot between two neurons.

    Parameters
    ----------
    spike_times_a : array-like
    spike_times_b : array-like
    bin_size : float
    duration : float, optional
    nperseg : int, optional
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if duration is None:
        a_max = float(np.max(spike_times_a)) if len(spike_times_a) > 0 else 1.0
        b_max = float(np.max(spike_times_b)) if len(spike_times_b) > 0 else 1.0
        duration = max(a_max, b_max) + bin_size

    bin_a, fs = spectral.spike_train_to_binary(spike_times_a, bin_size, duration)
    bin_b, _ = spectral.spike_train_to_binary(spike_times_b, bin_size, duration)

    freqs, coh = spectral.coherence(bin_a, bin_b, fs, nperseg)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=coh,
        mode="lines",
        line=dict(color="#d62728", width=1.5),
        hovertemplate="Freq: %{x:.1f} Hz<br>Coherence: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Frequency (Hz)"),
        yaxis=dict(title="Coherence", range=[0, 1]),
        template="plotly_white",
        width=900,
        height=400,
    )
    return fig


def plotly_band_power_comparison(spikes_by_neuron, bin_size=0.001,
                                  duration=None, bands=None,
                                  title="Band Power by Neuron"):
    """
    Grouped bar chart comparing power in frequency bands across neurons.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
    bin_size : float
    duration : float, optional
    bands : dict, optional
        Band name → (low_hz, high_hz).  Defaults to standard neuro bands.
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if bands is None:
        bands = spectral.STANDARD_BANDS

    neuron_ids = sorted(spikes_by_neuron.keys())
    band_names = list(bands.keys())

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2"]

    for b_idx, band_name in enumerate(band_names):
        low, high = bands[band_name]
        powers = []
        for nid in neuron_ids:
            binary, fs = spectral.spike_train_to_binary(
                spikes_by_neuron[nid], bin_size, duration
            )
            bp = spectral.band_power(binary, fs, low, high)
            powers.append(bp)

        fig.add_trace(go.Bar(
            name=f"{band_name} ({low}–{high} Hz)",
            x=[f"Neuron {n}" for n in neuron_ids],
            y=powers,
            marker_color=colors[b_idx % len(colors)],
            hovertemplate=(
                "%{x}<br>"
                f"{band_name}: " + "%{y:.2e}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Neuron"),
        yaxis=dict(title="Power"),
        barmode="group",
        template="plotly_white",
        width=900,
        height=450,
    )
    return fig
