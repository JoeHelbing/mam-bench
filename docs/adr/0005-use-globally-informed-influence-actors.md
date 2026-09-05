---
status: superseded
---

# Use globally informed typed Influence Actors

Superseded by [Schelling Influence Profile v2](../schelling-influence-profile-v2.md).
The body below records the retired v1 decision.

The first Model Evaluation replaces 16 fixed Ordinary Agent identities with 16 separate model-controlled Influence Actors: type A IDs `0..7` and type B IDs `150..157`. Their fixed exterior types remain visible to Ordinary Agents, but Influence Actors have no tolerance or satisfaction and are excluded from scored outcomes. All 16 receive global board visibility through bounded read tools and coordinate through a persistent free-text Coordination Board before choosing moves. This deliberately supersedes the earlier provisional partial-observation and bounded-communication direction: the first experiment asks whether a decentralized team can steer the whole system when information is not the bottleneck.

Each Model Evaluation runs for 20 Staged Rounds. Influence Actors may stay or move to any beginning-of-round vacancy, independent of distance or satisfaction. Their non-conflicting reservations receive priority in stable-ID order; Ordinary Agents then follow the frozen Schelling policy over remaining vacancies; every accepted move applies together. Matched integration and segregation cases optimize Ordinary Edge Homophily, with same-seed Counterfactual References masking the same 16 identities. Raw final Directional Lift is the primary score.

The locally running PydanticAI orchestrator calls `qwen/qwen3.8-27b` through OpenRouter, pinned to Phala with provider fallback disabled, and keeps a complete private audit history for each actor while replaying only a rolling three-round history to the model. Two synchronized waves each issue up to 16 concurrent requests: free-text coordination, then strict structured movement. Complete numeric trajectories and model evidence are retained in a versioned case directory. The implementation must pass deterministic fake-team tests and one live `(3/4, 25% empty, Seed 22, integration)` pilot before any full panel or ablation work.
