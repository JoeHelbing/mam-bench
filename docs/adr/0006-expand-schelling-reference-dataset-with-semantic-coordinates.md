# Expand the Schelling Reference Dataset with semantic coordinates

The second Schelling Reference Dataset uses a deliberately small Cartesian grid of board sizes 20, 60, and 100; all 23 behaviorally distinct tolerances; nine vacancy fractions spanning 1% through 50%; 50 seeds per cell; and a 30-transition cap. This gives enough replication, low-vacancy resolution, and board-size coverage for landscape analysis without the runaway cost of a dense sweep over every dimension. Full trajectories remain available because spatial dynamics matter to later analysis.

Cell identity and RNG entropy use exact board sizes and fractions rather than list indexes. Full artifacts therefore have semantic filenames and an artifact-only JSONL inventory, so reordering profile lists cannot change randomness or make the dataset unreadable. Schelling-specific Modal generation lives with the Schelling simulation package.

Full trajectories remain ignored developer analysis material because ordinary evaluation uses only a fixed initial state and Counterfactual Reference endpoint. Distributions package a compact evaluation fixture and the aggregate 621-cell landscape instead, avoiding a roughly 500 MiB wheel without discarding the evidence used to select the test condition.
