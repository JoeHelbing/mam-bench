---
status: accepted
---

# Select three fixed-vacancy final-satisfaction test spots

The candidate test panel uses three exact Landscape Cells selected from the board-20 slice of the v2 landscape. Each value is mean final satisfied-agent fraction across 50 Landscape Seeds. We use final satisfaction rather than equilibrium-only satisfaction so blocked and horizon-exhausted runs remain visible.

All three spots hold vacancy at `25%` while fractional preference varies:

- `1/2` preference: stable control with mean final satisfaction `1.000`, sample Seed SD `0.001`, and `98%` equilibrium.
- `3/4` preference: decision-sensitive transition with mean `0.762`, sample Seed SD `0.279`, `50%` equilibrium, `44%` blocked, and `6%` horizon-exhausted.
- `6/7` preference: deadlock stress test with mean `0.092`, sample Seed SD `0.036`, and `100%` blocked outcomes.

Holding vacancy constant avoids confounding model response with two simultaneous landscape changes. The `1/2` spot is primarily a non-regression control; the `3/4` spot should provide the strongest discrimination between steering policies; and the `6/7` spot tests whether model control can escape structural deadlock. The current benchmark implements only the `3/4` spot with held-out Evaluation Seed 50.

This replaces both the earlier provisional goal of approximately five Representative Regions and the mechanically selected global midpoint at `5/6` preference and `35%` empty. The global midpoint remains useful for reading the manifold but is weaker experimental design for an initial three-spot panel because vacancy changes with preference.
