"""
plotly_viz.py
-------------
Interactive Plotly-based visualizations for spike train data.

Every function returns a ``plotly.graph_objects.Figure`` that can be
displayed in Jupyter, Streamlit, or exported to standalone HTML.
"""

import numpy as np
import plotly.graph_objects as go

from . import analysis

# Consistent colour palette
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plotly_raster(spikes_by_neuron, title="Spike Raster Plot"):
    """
    Interactive raster plot — one row per neuron, one marker per spike.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
        Mapping from neuron ID to spike-time arrays.
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    neuron_ids = sorted(spikes_by_neuron.keys())

    for row, nid in enumerate(neuron_ids):
        times = spikes_by_neuron[nid]
        fig.add_trace(go.Scatter(
            x=times,
            y=[f"Neuron {nid}"] * len(times),
            mode="markers",
            marker=dict(
                symbol="line-ns-open",
                size=8,
                color=COLORS[row % len(COLORS)],
                line=dict(width=1.5),
            ),
            name=f"Neuron {nid}",
            hovertemplate=(
                "Neuron %{y}<br>"
                "Time: %{x:.4f} s<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time (s)",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title=""),
        template="plotly_white",
        width=950,
        height=max(300, 80 * len(neuron_ids)),
        showlegend=False,
    )
    return fig


def plotly_firing_rate(spikes_by_neuron, bin_size=0.5,
                       title="Population Firing Rate"):
    """
    Population-averaged firing rate over time with area fill.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
    bin_size : float
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bin_centers, rate_hz = analysis.population_rate(
        spikes_by_neuron, bin_size=bin_size
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=rate_hz,
        mode="lines",
        fill="tozeroy",
        line=dict(color="steelblue", width=2),
        hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.2f} Hz<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time (s)",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title="Firing rate (Hz)"),
        template="plotly_white",
        width=950,
        height=400,
    )
    return fig


def plotly_spike_histogram(spike_times, bin_size=0.5,
                            title="Spike Count Histogram"):
    """
    Histogram of spike counts per time bin for a single neuron.

    Parameters
    ----------
    spike_times : array-like
    bin_size : float
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bin_edges, counts = analysis.binned_spike_counts(spike_times, bin_size)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_edges[:-1],
        y=counts,
        width=bin_size * 0.9,
        marker_color="darkorange",
        marker_line=dict(color="black", width=0.5),
        hovertemplate=(
            "Bin: %{x:.2f}–%{customdata:.2f} s<br>"
            "Count: %{y}<extra></extra>"
        ),
        customdata=bin_edges[1:],
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Spike count"),
        template="plotly_white",
        width=950,
        height=400,
    )
    return fig


def plotly_isi_histogram(spike_times, bins=30,
                          title="ISI Distribution", log_scale=False):
    """
    Histogram of inter-spike intervals with optional log scale.

    Parameters
    ----------
    spike_times : array-like
    bins : int
    title : str
    log_scale : bool
        If True, use a log-scaled x-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    isis = analysis.inter_spike_intervals(spike_times)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=isis,
        nbinsx=bins,
        marker_color="mediumseagreen",
        marker_line=dict(color="black", width=0.5),
        hovertemplate="ISI: %{x:.4f} s<br>Count: %{y}<extra></extra>",
    ))

    # Mean ISI annotation
    if len(isis) > 0:
        mean_val = float(np.mean(isis))
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean ISI = {mean_val:.4f} s",
            annotation_position="top right",
        )

    xaxis_opts = dict(title="ISI (s)")
    if log_scale and len(isis) > 0:
        xaxis_opts["type"] = "log"

    fig.update_layout(
        title=title,
        xaxis=xaxis_opts,
        yaxis=dict(title="Count"),
        template="plotly_white",
        width=700,
        height=400,
    )
    return fig


def plotly_firing_rate_comparison(spikes_by_neuron, duration=None,
                                   title="Firing Rate by Neuron"):
    """
    Bar chart comparing average firing rate across neurons.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
    duration : float, optional
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    rates = analysis.firing_rates_all(spikes_by_neuron, duration=duration)
    neuron_ids = sorted(rates.keys())
    values = [rates[n] for n in neuron_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Neuron {n}" for n in neuron_ids],
        y=values,
        marker_color="slateblue",
        marker_line=dict(color="black", width=0.5),
        hovertemplate="Neuron %{x}<br>Rate: %{y:.2f} Hz<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Neuron"),
        yaxis=dict(title="Firing rate (Hz)"),
        template="plotly_white",
        width=700,
        height=400,
    )
    return fig


def plotly_population_heatmap(spikes_by_neuron, bin_size=0.5,
                               title="Population Activity Heatmap"):
    """
    Neuron × time-bin heatmap of firing rates.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
    bin_size : float
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    neuron_ids = sorted(spikes_by_neuron.keys())

    # Compute firing rate for each neuron
    rate_matrix = []
    bin_centers = None
    for nid in neuron_ids:
        centers, rate_hz = analysis.binned_firing_rate(
            spikes_by_neuron[nid], bin_size=bin_size
        )
        rate_matrix.append(rate_hz)
        if bin_centers is None:
            bin_centers = centers

    # Pad rows to same length (in case durations differ slightly)
    max_len = max(len(r) for r in rate_matrix)
    for i in range(len(rate_matrix)):
        if len(rate_matrix[i]) < max_len:
            rate_matrix[i] = np.pad(
                rate_matrix[i], (0, max_len - len(rate_matrix[i]))
            )

    z = np.array(rate_matrix)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=np.round(bin_centers[:max_len], 2) if bin_centers is not None else None,
        y=[f"Neuron {n}" for n in neuron_ids],
        colorscale="YlOrRd",
        colorbar=dict(title="Rate (Hz)"),
        hovertemplate=(
            "Neuron: %{y}<br>"
            "Time: %{x:.2f} s<br>"
            "Rate: %{z:.2f} Hz<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title=""),
        template="plotly_white",
        width=950,
        height=max(300, 50 * len(neuron_ids)),
    )
    return fig
