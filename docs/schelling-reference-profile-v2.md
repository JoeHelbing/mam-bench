# Schelling Reference Dataset v2

This document freezes the ordinary-agent mechanics, parameter grid, randomness, and artifact format used by Schelling Reference Dataset v2.

## Parameter grid

Every Landscape Cell is one exact combination of:

- board size: `20`, `60`, or `100`;
- one of the 23 behaviorally distinct radius-1 Moore-neighborhood tolerances;
- vacancy fraction: `1/100`, `1/40`, `1/20`, `1/10`, `1/5`, `1/4`, `3/10`, `2/5`, or `1/2`.

Each cell contains Landscape Seeds `0..49`:

```text
3 board sizes x 23 tolerances x 9 vacancy levels = 621 artifacts
621 artifacts x 50 Landscape Seeds = 31,050 runs
```

Vacancy counts are exact: `board_size² × vacancy_fraction`. The remaining cells split evenly between type A and type B.

## Ordinary-agent mechanics

Boards are square tori. Each occupied cell has the eight radius-1 Moore neighbors, wrapping across edges. An agent is satisfied when it has no occupied neighbors or when:

```text
same_type_neighbors / occupied_neighbors >= tolerance
```

The implementation compares exact integers rather than floating-point ratios.

Each Staged Round:

1. evaluates satisfaction against one frozen board;
2. visits unhappy agents in a seeded random order;
3. reserves for each agent the nearest still-available vacancy that would satisfy it against the frozen board;
4. breaks true nearest-distance ties with a separate random stream; and
5. applies all reserved moves together.

A run stops at equilibrium, when blocked, or after 30 transitions. It retains initialization plus every completed transition, so its trajectory length is between 1 and 31.

## Randomness

NumPy `SeedSequence` and `PCG64` receive two public master words spelling `MAMBENCH`, a stream ID, semantic cell coordinates, and the Landscape Seed.

Every stream includes:

- board size;
- vacancy numerator and denominator;
- stream ID; and
- seed ID.

Reservation-order and tie-break streams also include tolerance numerator and denominator. Initialization excludes tolerance so runs with the same board size, vacancy fraction, and seed begin from identical paired boards across all tolerances.

Coordinate list indexes are not part of the RNG contract. Reordering a list cannot silently change generated data.

## Full trajectory artifacts

The developer-only full dataset stores one compressed NumPy archive per Landscape Cell with a semantic filename:

```text
cells/board-020__tolerance-3-of-4__vacancy-1-of-4.npz
```

Each archive contains:

- `cell_types`: `uint8[50,31,board_size,board_size]`;
- `agent_locations`: `uint16[50,31,n_agents]`;
- `agent_types`: `uint8[n_agents]`;
- `trajectory_lengths`: `uint16[50]`;
- `terminal_status`: `uint8[50]`;
- `landscape_seed_ids`: `uint16[50]`; and
- scalar board size, tolerance fraction, vacancy fraction, and vacancy count.

Cell-type values are `0` for empty, `1` for type A, and `2` for type B. Unused padded states use cell value `255` and location value `65535`. NPZ loading always uses `allow_pickle=False`.

## JSONL artifact inventory

`manifest.jsonl` contains exactly one compact JSON object per NPZ artifact. A record has this shape:

```json
{"board_size":20,"tolerance":"3/4","vacancy_fraction":"1/4","vacancy_count":100,"run_count":50,"path":"cells/board-020__tolerance-3-of-4__vacancy-1-of-4.npz","byte_size":88717,"sha256":"..."}
```

The inventory contains no global header, profile copy, coordinate indexes, array-schema copy, timestamp, or software-version block. Fixed scientific rules live in code and this document. Validation requires exactly the 621 canonical records, matching hashes and sizes, expected typed arrays and padding, exact populations, seeds `0..49`, and tolerance-paired initial boards. The simulation mechanics themselves are covered by focused tests rather than replay-validating every generated state.

The 540 MiB full dataset lives under ignored `.scratch/schelling-reference-v2-full/`. It supports development and scientific analysis but is not packaged.

## Packaged products

`src/mam_bench/data/schelling-reference-v2/` contains only:

- `landscape.jsonl`: 621 per-cell aggregate outcome records;
- `evaluation-reference.json`: readable fixed-case parameters, comparator metrics, and the NPZ hash; and
- `evaluation-reference.npz`: Seed 50 initial grid, stable locations and types, and terminal Counterfactual Reference grid and locations.

These files are sufficient to explain Test Spot selection, initialize the fixed evaluation, expose its reference endpoint to Influence Actors, and score against the frozen comparator. Intermediate Counterfactual Reference states are not used by evaluation.

## Generation

Schelling owns its Modal adapter:

```bash
modal run src/mam_bench/simulations/schelling/utils/modal_reference_sweep.py \
  --output-dir .scratch/schelling-reference-v2-full \
  --receipt-path .scratch/schelling-reference-v2-receipt.json
```

The receipt is resumable working state. `manifest.jsonl` is written only after all 621 full artifacts arrive. Final validation and compact publication run through:

```bash
uv run src/mam_bench/simulations/schelling/utils/publish_reference_data.py
```

## Influence benchmark integration

The existing `schelling-influence-pilot-v1` simulation selects the v2 cell with board size 20, tolerance `3/4`, and vacancy `1/4`. Landscape Seeds are `0..49`, so held-out Evaluation Seed `50` supplies its same-seed Counterfactual Reference.
