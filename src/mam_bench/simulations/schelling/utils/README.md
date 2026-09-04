# Schelling reference-data utilities

This directory contains offline developer tools. Normal benchmark runs do not import them.

## Files

- `modal_reference_sweep.py` runs the 621-cell, 50-seed reference sweep on Modal and stores full trajectories under `.scratch/`.
- `dataset.py` builds and validates the full trajectory archives and their manifest.
- `analysis.py` reduces the full trajectories to one aggregate record per Landscape Cell.
- `reference_data.py` builds and writes the fixed Seed 50 evaluation fixture.
- `publish_reference_data.py` runs validation, analysis, and fixture generation to refresh the three compact files shipped in `mam_bench.data`.

## Regenerate reference data

From the repository root:

```bash
modal run src/mam_bench/simulations/schelling/utils/modal_reference_sweep.py \
  --output-dir .scratch/schelling-reference-v2-full \
  --receipt-path .scratch/schelling-reference-v2-receipt.json

uv run src/mam_bench/simulations/schelling/utils/publish_reference_data.py
```

The first command is resumable. The second validates the completed sweep and rewrites:

- `src/mam_bench/data/schelling-reference-v2/landscape.jsonl`
- `src/mam_bench/data/schelling-reference-v2/evaluation-reference.json`
- `src/mam_bench/data/schelling-reference-v2/evaluation-reference.npz`

Do not run either command during ordinary benchmark execution.
