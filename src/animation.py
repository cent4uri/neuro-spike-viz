"""
animation.py
------------
Animated spike-train playback using Plotly frames.

Provides a sliding time window animation over the raster plot
with Play/Pause controls and a manual time scrubber.
"""

import numpy as np
import plotly.graph_objects as go


# Consistent colour palette (same as plotly_viz)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def create_spike_animation(
    spikes_by_neuron,
    window_size=2.0,
    step=0.1,
    fps=10,
    title="Spike Train Animation",
):
    """
    Create an animated raster plot with a sliding time window.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
        Mapping of neuron ID to spike-time arrays.
    window_size : float, default=2.0
        Width of the visible time window in seconds.
    step : float, default=0.1
        Time step between animation frames in seconds.
    fps : int, default=10
        Frames per second for playback speed.
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
        Animated figure with Play/Pause buttons and slider.
    """
    neuron_ids = sorted(spikes_by_neuron.keys())
    n_neurons = len(neuron_ids)

    # Determine total duration
    all_times = np.concatenate([
        t for t in spikes_by_neuron.values() if len(t) > 0
    ]) if any(len(t) > 0 for t in spikes_by_neuron.values()) else np.array([0])
    total_duration = float(all_times.max())

    # Compute frame start times
    frame_starts = np.arange(0, total_duration - window_size + step, step)

    # Limit total frames for performance
    max_frames = 200
    if len(frame_starts) > max_frames:
        step = (total_duration - window_size) / max_frames
        frame_starts = np.arange(0, total_duration - window_size + step, step)
        frame_starts = frame_starts[:max_frames]

    y_labels = [f"Neuron {n}" for n in neuron_ids]

    # Build initial data (first window)
    initial_traces = _raster_traces_in_window(
        spikes_by_neuron, neuron_ids, 0, window_size
    )

    # Build frames
    frames = []
    slider_steps = []

    for i, t_start in enumerate(frame_starts):
        t_end = t_start + window_size
        frame_traces = _raster_traces_in_window(
            spikes_by_neuron, neuron_ids, t_start, t_end
        )

        frames.append(go.Frame(
            data=frame_traces,
            name=str(i),
            layout=go.Layout(
                xaxis=dict(range=[t_start, t_end]),
                annotations=[dict(
                    x=0.5, y=1.08, xref="paper", yref="paper",
                    text=f"Window: {t_start:.2f} – {t_end:.2f} s",
                    showarrow=False, font=dict(size=12),
                )],
            ),
        ))

        slider_steps.append(dict(
            args=[[str(i)], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{t_start:.1f}",
            method="animate",
        ))

    # Create figure
    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=title,
            xaxis=dict(
                title="Time (s)",
                range=[0, window_size],
            ),
            yaxis=dict(
                title="",
                categoryorder="array",
                categoryarray=y_labels[::-1],
            ),
            template="plotly_white",
            width=950,
            height=max(350, 70 * n_neurons),
            showlegend=False,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.0,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(
                                duration=int(1000 / fps),
                                redraw=True,
                            ),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                x=0.05,
                len=0.9,
                xanchor="left",
                y=-0.05,
                currentvalue=dict(
                    prefix="Time: ",
                    suffix=" s",
                    visible=True,
                ),
                transition=dict(duration=0),
            )] if slider_steps else [],
        ),
        frames=frames,
    )

    return fig


def _raster_traces_in_window(spikes_by_neuron, neuron_ids, t_start, t_end):
    """Build scatter traces for spikes within a time window."""
    traces = []
    for row, nid in enumerate(neuron_ids):
        times = spikes_by_neuron[nid]
        mask = (times >= t_start) & (times <= t_end)
        visible_times = times[mask]

        traces.append(go.Scatter(
            x=visible_times,
            y=[f"Neuron {nid}"] * len(visible_times),
            mode="markers",
            marker=dict(
                symbol="line-ns-open",
                size=10,
                color=COLORS[row % len(COLORS)],
                line=dict(width=2),
            ),
            hovertemplate=(
                f"Neuron {nid}<br>"
                "Time: %{x:.4f} s<extra></extra>"
            ),
        ))

    return traces


def create_firing_rate_animation(
    spikes_by_neuron,
    bin_size=0.5,
    window_size=5.0,
    step=0.5,
    fps=5,
    title="Firing Rate Animation",
):
    """
    Animated population firing rate with a sliding highlight window.

    Shows the full firing-rate curve with a moving shaded band
    highlighting the current time window.

    Parameters
    ----------
    spikes_by_neuron : dict[int, np.ndarray]
    bin_size : float
    window_size : float
    step : float
    fps : int
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from . import analysis

    bin_centers, rate_hz = analysis.population_rate(
        spikes_by_neuron, bin_size=bin_size
    )

    if len(bin_centers) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    total_duration = float(bin_centers[-1])
    frame_starts = np.arange(0, total_duration - window_size + step, step)

    max_frames = 200
    if len(frame_starts) > max_frames:
        step = (total_duration - window_size) / max_frames
        frame_starts = np.arange(0, total_duration - window_size + step, step)
        frame_starts = frame_starts[:max_frames]

    max_rate = float(np.max(rate_hz)) * 1.1

    # Initial frame
    t0 = frame_starts[0] if len(frame_starts) > 0 else 0
    initial_shapes = [_highlight_rect(t0, t0 + window_size, max_rate)]

    frames = []
    slider_steps = []

    for i, t_start in enumerate(frame_starts):
        t_end = t_start + window_size

        frames.append(go.Frame(
            name=str(i),
            layout=go.Layout(
                shapes=[_highlight_rect(t_start, t_end, max_rate)],
                annotations=[dict(
                    x=0.5, y=1.08, xref="paper", yref="paper",
                    text=f"Window: {t_start:.1f} – {t_end:.1f} s",
                    showarrow=False, font=dict(size=12),
                )],
            ),
        ))

        slider_steps.append(dict(
            args=[[str(i)], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{t_start:.1f}",
            method="animate",
        ))

    fig = go.Figure(
        data=[go.Scatter(
            x=bin_centers,
            y=rate_hz,
            mode="lines",
            fill="tozeroy",
            line=dict(color="steelblue", width=2),
            fillcolor="rgba(70,130,180,0.15)",
            hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.2f} Hz<extra></extra>",
        )],
        layout=go.Layout(
            title=title,
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Firing rate (Hz)", range=[0, max_rate]),
            shapes=initial_shapes,
            template="plotly_white",
            width=950,
            height=400,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.0,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(
                                duration=int(1000 / fps),
                                redraw=True,
                            ),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                x=0.05,
                len=0.9,
                xanchor="left",
                y=-0.05,
                currentvalue=dict(
                    prefix="Time: ",
                    suffix=" s",
                    visible=True,
                ),
                transition=dict(duration=0),
            )] if slider_steps else [],
        ),
        frames=frames,
    )

    return fig


def _highlight_rect(t_start, t_end, y_max):
    """Return a Plotly shape dict for the highlight band."""
    return dict(
        type="rect",
        x0=t_start,
        x1=t_end,
        y0=0,
        y1=y_max,
        fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="darkorange", width=1.5, dash="dot"),
        layer="above",
    )
