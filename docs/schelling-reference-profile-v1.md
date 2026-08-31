# Schelling Reference Profile v1

This document freezes the ordinary-agent system used to generate MAM-Bench's first Reference Landscape. It defines one modernized Schelling model and its raw dataset contract; model-controlled agents, steering objectives, derived segregation metrics, and Representative Region selection are out of scope.

## Historical callout

Schelling's 1971 checkerboard experiments described multiple preference formulations and commonly used bounded boards, sequential movement, an eight-cell neighborhood, and movement toward a nearest satisfactory vacancy. This profile keeps the fractional local preference, eight-cell neighborhood, stationary satisfied agents, and nearest satisfactory relocation, but deliberately uses a torus and staged rounds. It is therefore the Schelling Reference Profile, not the original or canonical Schelling model.

## Fixed system

- Grid: 20x20.
- Topology: toroidal in both dimensions.
- Cell states: empty, type A, or type B.
- Groups: two symmetric groups with equal counts.
- Neighborhood: radius-1 Moore neighborhood, excluding the focal cell.
- Maximum transitions: 500.
- Satisfied agents stay in place.
- Agents with zero occupied neighbors are satisfied.

## Exact tolerance rule

Tolerance is an exact nonnegative rational `p/q`. An agent with at least one occupied neighbor is satisfied exactly when:

```text
same_type_neighbors * q >= occupied_neighbors * p
```

The implementation must not use floating-point comparisons for satisfaction.

The landscape uses the 23 behaviorally distinct fractions available with one to eight occupied neighbors, in this order:

```text
0/1,
1/8, 1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 3/8, 2/5, 3/7,
1/2,
4/7, 3/5, 5/8, 2/3, 5/7, 3/4, 4/5, 5/6, 6/7, 7/8,
1/1
```

These are the Farey sequence of order 8. Decimal values are descriptive only and never determine behavior.

## Vacancy levels and initialization

The landscape uses these exact vacancy fractions:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
```

On the 400-cell grid they produce 40, 60, 80, 100, 120, 140, and 160 vacancies. The remaining cells are split exactly and evenly between types A and B.

Each run assigns stable agent IDs `0..n_agents-1`. The first half are type A and the second half type B. Initialization uniformly shuffles the complete set of agent IDs and empty tokens across the 400 cells. At a fixed vacancy level and Landscape Seed, every tolerance uses the same initialized grid.

## Deterministic randomness

Randomness uses NumPy `SeedSequence` and `PCG64`. The fixed public master seed words are:

```text
0x4D414D42, 0x454E4348  # ASCII "MAMBENCH"
```

No seed derivation may use Python's `hash()`.

The stream identifiers are:

```text
0 = initialization
1 = reservation order
2 = equal-distance tie breaking
```

For vacancy index `v`, tolerance index `t`, and seed ID `s`, construct streams from these entropy words:

```text
initialization:    [master_0, master_1, 0, v, s]
reservation order: [master_0, master_1, 1, v, t, s]
tie breaking:      [master_0, master_1, 2, v, t, s]
```

The separate dynamics streams prevent a conditional tie-break draw from changing later reservation-order shuffles. The tie-breaking stream advances only when two or more equal-distance nearest candidates exist; selecting a sole nearest candidate consumes no tie-breaking draw. Landscape Seed IDs are `0..19`. Held-out Evaluation Seed IDs begin at `20` and do not contribute to the Reference Landscape.

## Staged round

A state transition follows this exact sequence:

1. Freeze the current cell-type grid and agent locations.
2. Evaluate every agent's satisfaction from that state.
3. If every agent is satisfied, terminate at equilibrium.
4. Sort unhappy agent IDs ascending, then uniformly permute them with the reservation-order stream.
5. In that order, let each unhappy agent seek a destination among the vacancies that existed at the beginning of the round and remain unreserved.
6. A candidate is satisfactory when the moving agent would satisfy the exact tolerance rule in the frozen state with only its own origin treated as empty and the candidate occupied by its type. Other proposed moves are not reflected in this calculation.
7. Select candidates at minimum toroidal Chebyshev distance. When two or more nearest candidates remain, break that tie uniformly with one tie-breaking-stream draw; a sole nearest candidate consumes no draw. Reserve the selected vacancy. If no satisfactory unreserved vacancy exists, the agent stays.
8. If the complete reservation pass produces no reservations, terminate as blocked.
9. Apply every reserved move together. Origins become vacancies only after application and cannot be selected during the same round.
10. Append the resulting state and increment `rounds_completed`.

For positions `(r1, c1)` and `(r2, c2)`, toroidal Chebyshev distance is:

```text
dr = min(abs(r1 - r2), 20 - abs(r1 - r2))
dc = min(abs(c1 - c2), 20 - abs(c1 - c2))
distance = max(dr, dc)
```

Candidate vacancies are enumerated in row-major order before distance filtering and random tie selection.

## Terminal outcomes and indexing

State index 0 is the initialized state. Only an applied reservation batch creates another state. Consequently:

```text
trajectory_length = rounds_completed + 1
1 <= trajectory_length <= 501
```

A blocked attempt does not create a duplicate state or increment `rounds_completed`.

After the 500th applied transition, classify the resulting state in this priority order:

1. `equilibrium`: every agent is satisfied;
2. `blocked`: unhappy agents remain and none has any satisfactory beginning-of-round vacancy;
3. `horizon_exhausted`: at least one further move remains possible.

## Reference Landscape

The Cartesian product contains:

```text
23 tolerances x 7 vacancy levels = 161 Landscape Cells
161 cells x 20 Landscape Seeds = 3,220 runs
```

The 20 initialized grids at each vacancy level are paired across all tolerance values.

## Raw trajectory artifacts

Generation produces one compressed NumPy `.npz` archive per Landscape Cell. An NPZ file is a ZIP container of named, typed NumPy arrays. Loading must use `allow_pickle=False`.

Each cell artifact contains:

- `cell_types`: `uint8[20, 501, 20, 20]`
- `agent_locations`: `uint16[20, 501, n_agents]`
- `agent_types`: `uint8[n_agents]`
- `trajectory_lengths`: `uint16[20]`
- `terminal_status`: `uint8[20]`
- `landscape_seed_ids`: `uint16[20]`
- scalar exact parameter indices, numerators, denominators, and vacancy count

Cell-type codes are:

```text
0 = empty
1 = type A
2 = type B
255 = unused state padding
```

Each agent location is the flattened cell index `row * 20 + column`. Location code `65535` marks unused state padding. Agent ID is the final `agent_locations` axis index. Terminal-status codes and every other code mapping are defined in the manifest.

Entries at state indices greater than or equal to a run's `trajectory_length` contain only the relevant padding sentinel. The stored cell-type grid and agent locations must agree at every retained state; generation validation fails closed on any mismatch.

## Manifest and provenance

A Pydantic-validated JSON manifest records:

- dataset and schema versions;
- the complete fixed profile;
- exact ordered tolerance and vacancy coordinates;
- RNG algorithm, master seed, stream IDs, and derivation contract;
- array names, shapes, dtypes, sentinels, and code mappings;
- every expected cell artifact, byte size, and SHA-256 digest;
- engine, Python, NumPy, Pydantic, and Modal client versions;
- generation timestamp and completion counts.

The manifest is complete only after all 161 artifacts pass schema, hash, state-consistency, exact-count, and trajectory-length validation. Derived segregation metrics and region-selection features are intentionally absent from Reference Landscape v1 and will be specified after raw generation.

## Modal execution

The host-neutral engine is tested locally before cloud execution. Modal receives 161 independent CPU tasks, one per Landscape Cell; each task runs its 20 seeds and returns one compressed artifact. The local runner writes each artifact atomically, validates it, and can skip already valid cells when resuming. The final dataset lives under `data/reference-landscape/v1/`; Modal function results or remote storage are not the sole durable copy.
