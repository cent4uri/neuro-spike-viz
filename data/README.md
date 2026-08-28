# Data

## `sample_spikes.csv`

A small synthetic spike-train dataset (5 neurons, 10 seconds) generated
with independent Poisson processes at randomized firing rates. It exists
so the project runs out of the box with no external downloads.

Format:

```
neuron_id,spike_time
1,0.0123
1,0.0841
2,0.0547
...
```

- `neuron_id`: integer identifier for the neuron
- `spike_time`: time of the spike, in seconds, relative to recording start

## Using your own data

Any CSV with the two columns above will work with `src/loader.py`.

## Public datasets

If you'd like real neural recordings instead of synthetic data:

- **Allen Brain Atlas / Allen Brain Observatory** — https://portal.brain-map.org/
  Large-scale physiology datasets from the Allen Institute; many are
  distributed as NWB files (see "Future Improvements" in the main README).
- **CRCNS (Collaborative Research in Computational Neuroscience)** —
  https://crcns.org/ — a long-running data-sharing archive of neurophysiology
  recordings across many labs and species (requires free registration).

To use these, export or convert spike times to the `neuron_id,spike_time`
CSV format above, then point `loader.load_spike_csv()` at the file.
