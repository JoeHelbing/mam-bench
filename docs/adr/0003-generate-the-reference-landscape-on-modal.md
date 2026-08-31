---
status: accepted
---

# Generate the reference landscape on Modal

The complete 3,220-run Reference Landscape will execute on Modal's serverless CPU platform and retain every 20x20 cell-type grid from initialization through termination, not only endpoints or aggregate round metrics. Each state also records one location per stable run-local agent ID, while a static table in each cell artifact maps those IDs to types; the dataset does not duplicate the grid as an occupant-ID grid. Modal can fan out this independent batch cheaply, and the full trajectories preserve spatial dynamics for later regime analysis and audit. The simulation engine remains host-neutral and locally testable; Modal is an execution adapter, not part of the Schelling semantics. Results must be downloaded into a versioned local dataset with a manifest rather than left solely in ephemeral function results or remote storage. The generation dataset will contain raw trajectories plus the parameters, seed identity, trajectory length, and terminal reason required to interpret them; derived segregation metrics and the region-selection analysis are intentionally deferred until after generation. Artifacts use one compressed NumPy NPZ file per Landscape Cell plus a Pydantic JSON manifest, as specified in `docs/schelling-reference-profile-v1.md`.
