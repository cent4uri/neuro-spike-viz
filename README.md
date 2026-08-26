# Neuro Spike Visualizer (NSV)

A Python-based neuroscience data analysis and visualization project for exploring neuronal spike trains, measuring firing activity, and identifying temporal patterns in neural data.

> **Project Status:** In Progress
> Core spike-train loading, analysis, and visualization features are currently being developed.

## Overview

Neuro Spike Visualizer provides a simple workflow for working with neuronal spike-train data. It converts spike timestamps into interpretable visualizations and basic firing-rate measurements, making it easier to explore neural activity across individual and multiple neurons.

## Features

* Visualize neuronal spike trains over time
* Generate spike raster plots
* Calculate neuron firing rates
* Analyze spike counts and temporal activity
* Explore inter-spike interval (ISI) distributions
* Compare activity across multiple neurons
* Perform interactive analysis with Jupyter Notebook
* Create scientific visualizations using Matplotlib

## Technologies

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Data analysis and project development    |
| NumPy            | Numerical and array-based computation    |
| Matplotlib       | Scientific visualization                 |
| Jupyter Notebook | Interactive analysis and experimentation |

## Installation

Clone the repository:

```bash
git clone https://github.com/cent4uri/neuro-spike-visualizer.git
cd neuro-spike-visualizer
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/spike_visualization.ipynb
```

## Quick Start

The project can also be used directly from Python:

```python
from src import loader, analysis, visualization
import matplotlib.pyplot as plt

spikes = loader.load_spike_csv("data/sample_spikes.csv")

visualization.plot_raster(spikes)
plt.show()

visualization.plot_firing_rate_over_time(spikes, bin_size=0.5)
plt.show()

rates = analysis.firing_rates_all(spikes)
print(rates)
```

## Visualizations

The project currently supports analysis and visualization of:

* **Spike Raster Plot** — displays individual spike events across neurons and time.
* **Firing Rate Over Time** — shows changes in neuronal activity across time windows.
* **Spike Count Histogram** — summarizes spike activity within defined intervals.
* **ISI Distribution** — explores the time intervals between consecutive spikes.
* **Multi-Neuron Comparison** — compares firing activity across multiple neurons.

## Dataset

The repository includes a small synthetic dataset:

```text
data/sample_spikes.csv
```

This allows the project to run immediately without requiring external downloads.

The project can also work with publicly available neuroscience datasets, including:

* **Allen Brain Atlas**
* **CRCNS — Collaborative Research in Computational Neuroscience**

Example spike-train format:

```text
neuron_id,spike_time
1,0.012
1,0.083
1,0.147
2,0.054
2,0.210
```

See `data/README.md` for information about preparing external datasets for use with the project.

## Learning Objectives

This project demonstrates practical experience with:

* Neural spike-train analysis
* Firing-rate computation
* Scientific data visualization
* Time-series exploration
* Python data-analysis workflows
* Working with neuroscience datasets
* Basic computational neuroscience concepts

## Future Improvements

Planned improvements include:

* Interactive Plotly visualizations
* NWB file support
* Animated spike-train playback
* Spectral and frequency-domain analysis
* Spike-sorting integration
* Streamlit-based interactive interface
* Additional neuroscience datasets

## References

* Allen Institute for Brain Science
* CRCNS Data Sharing Initiative
* NumPy Documentation
* Matplotlib Documentation
* Jupyter Documentation

## Acknowledgements

This project builds on publicly available neuroscience datasets and the open-source Python scientific-computing ecosystem.

Developed as a practical project for exploring **neural data analysis, scientific visualization, and computational neuroscience**.

> **Note:** This project uses publicly available and synthetic datasets for educational and research-oriented analysis.
