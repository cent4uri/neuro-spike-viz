# Neuro Spike Visualizer (NSV)

A Python-based data analysis and visualization project for exploring neuronal spike trains, measuring neural firing activity, and understanding temporal patterns in spike-train data.

Neuro Spike Visualizer (NSV) provides a beginner-friendly workflow for working with neuronal spike data, from loading and analyzing spike timestamps to generating scientific visualizations.

> ** Project Status: In Progress**
>
> The core spike-train loading, analysis, and visualization components are being developed incrementally.

## Features

- Visualize neuronal spike trains over time
- Compute and display neuron firing rates
- Generate raster plots and spike histograms
- Explore activity from different neurons
- Interactive analysis with Jupyter Notebook
- Clean Matplotlib visualizations

## Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/cent4uri/neuro-spike-visualizer.git
cd neuro-spike-visualizer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/spike_visualization.ipynb
```

## Quick start (without Jupyter)

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

## Example Visualizations

- Spike raster plot
- Firing rate over time
- Spike count histogram
- Inter-spike interval (ISI) distribution
- Multi-neuron firing rate comparison

## Datasets

This project ships with a small synthetic dataset (`data/sample_spikes.csv`)
so it runs immediately with no downloads. It also works with real,
publicly available neuroscience datasets such as:

- **Allen Brain Atlas** — https://portal.brain-map.org/
- **CRCNS** (Collaborative Research in Computational Neuroscience) — https://crcns.org/

You may also use your own spike train CSV files. Example format:

```
neuron_id,spike_time
1,0.012
1,0.083
1,0.147
2,0.054
2,0.210
```

See `data/README.md` for more on converting external datasets to this format.

## Learning Objectives

This project helps you learn:

- Neural spike train analysis
- Firing rate computation
- Scientific visualization
- Basic computational neuroscience
- Working with neuroscience datasets
- Python data analysis workflows

## Future Improvements

- Interactive Plotly visualizations
- Support for NWB (Neurodata Without Borders) files
- Animated spike playback
- Spectral analysis
- Spike sorting integration
- GUI using Streamlit

## References

- Allen Institute for Brain Science — https://portal.brain-map.org/
- CRCNS Data Sharing Initiative — https://crcns.org/
- NumPy Documentation — https://numpy.org/
- Matplotlib Documentation — https://matplotlib.org/
- Jupyter Documentation — https://jupyter.org/

## Acknowledgements

Special thanks to:

- Allen Institute for Brain Science
- CRCNS
- Open-source Python scientific computing community
- NumPy, Matplotlib, and Jupyter developers

Beginner-friendly project for learning computational neuroscience through
data visualization and neural signal analysis.

Note: This project uses publicly available datasets
