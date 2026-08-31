---
status: accepted
---

# Map behaviorally distinct tolerance-vacancy cells

The first Reference Landscape will vary global tolerance and vacancy fraction while fixing two symmetric groups at 50/50. Tolerance uses the 23 behaviorally distinct fractions available with one to eight occupied neighbors (the Farey sequence of order 8), rather than an arbitrary decimal grid; vacancy uses 0.10 through 0.40 in increments of 0.05. Each 20x20 initial grid has exact vacancy and group counts placed by a uniform shuffle. The resulting 161 Landscape Cells each receive 20 seeds, with initial grids paired across tolerance values at each vacancy level, for 3,220 ordinary-agent runs. This replication estimates stochastic outcome distributions before region selection; the three later seeds per Representative Region are held out and do not define the regions.
