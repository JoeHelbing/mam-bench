---
status: accepted
---

# Select three fixed-vacancy final-satisfaction test spots

The first test panel uses three exact Landscape Cells selected from a 23x7 surface whose z value is mean final satisfied-agent fraction across all 20 Landscape Seeds. We use final satisfaction rather than equilibrium-only satisfaction so blocked and horizon-exhausted runs remain visible.

All three spots hold vacancy at `25%` while fractional preference varies:

- `1/2` preference: stable control with mean final satisfaction `1.000`, sample Seed SD `0.000`, and `100%` equilibrium.
- `3/4` preference: decision-sensitive transition with mean `0.717`, sample Seed SD `0.298`, and a `50%` equilibrium / `50%` blocked split.
- `6/7` preference: deadlock stress test with mean `0.100`, sample Seed SD `0.042`, and `100%` blocked outcomes.

Holding vacancy constant avoids confounding model response with two simultaneous landscape changes. The `1/2` spot is primarily a non-regression control; the `3/4` spot should provide the strongest discrimination between steering policies; and the `6/7` spot tests whether model control can escape structural deadlock. Later tests use held-out Evaluation Seed IDs 20, 21, and 22 at each spot.

This replaces both the earlier provisional goal of approximately five Representative Regions and the mechanically selected global midpoint at `5/6` preference and `35%` empty. The global midpoint remains useful for reading the manifold but is weaker experimental design for an initial three-spot panel because vacancy changes with preference.
