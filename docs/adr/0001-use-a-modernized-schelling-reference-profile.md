---
status: accepted
---

# Use a modernized Schelling reference profile

The first Reference Landscape will use one fixed, modernized Schelling profile: a 20x20 toroidal grid, staged rounds, a radius-1 Moore neighborhood, fractional tolerance over occupied neighbors, stationary satisfied agents, and dissatisfied agents seeking the nearest satisfactory vacancy. An agent with zero occupied neighbors is satisfied. Runs end at equilibrium, a blocked state, or a 500-round horizon. Each unhappy agent evaluates candidate vacancies against the beginning-of-round state while treating its origin as vacant. In a seeded random order, unhappy agents choose the nearest still-unreserved satisfactory vacancy; all reserved moves are then applied together. Nearest vacancies are determined by toroidal Chebyshev distance, with equal-distance ties broken uniformly at random. This deliberately differs from Schelling's bounded, sequential checkerboard experiments: toroidal geometry removes edge effects, and staged rounds suit a performant native-CPU generator. We will describe it as the Schelling Reference Profile, not as the original or canonical Schelling model. Sequential reservation therefore affects destination allocation, while satisfaction evaluation and state mutation remain staged.
