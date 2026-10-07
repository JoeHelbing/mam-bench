# Rules and scores

Both worlds start from the same seed and use the same staged turn mechanism:
selected replacement identities decide first, then the remaining ordinary agents.
Model decisions can overlap up to the configured concurrency limit, but claims
are accepted in seeded priority order, not response order. Ordinary decisions
use that same priority and see earlier reservations. Within each phase, physical
actions settle only after both groups finish. Invalid claims retain their turn for bounded
retries or fallback; a slow earlier turn can delay later claims.

Turn priorities and action randomness have separate, stable addresses: case seed,
step, role phase, identity, and purpose. Replacing an ordinary decision leaves its
random slot unused instead of consuming a proposal in a different world state.
Jail, release, or early termination in one world cannot shift another agent's
random stream. Different available actions can still map the same randomness to
different choices. Model outputs and message timing are not made deterministic.

This changes the ordinary reference dynamics. Historical calibration archives
remain historical evidence, not interchangeable baselines for this scheduler.
The shipped cases use the current-rules matched-schedule scripted calibration.

Schelling preserves strict local improvement for ordinary agents: satisfaction
uses occupied radius-one neighbors and a floating-point tolerance; unhappy
agents choose the nearest predicted improvement within their vision, with random
ties. Destination predictions use only cells visible from the current origin,
excluding the origin itself. Unknown cells are excluded; zero known occupied
neighbors predicts quality one. Model agents may request any starting vacancy.
Selected identities reserve first in both worlds, followed by the remaining
ordinary identities; both groups use seeded priorities and all moves settle
together. Ordinary execution stops at equilibrium, blockage, or the horizon;
controlled execution runs to the configured horizon.

Each model-controlled Schelling agent receives a JSON observation with its step,
location, reference-final and current homophily, and remaining vacancy count.
Visible ordinary locations are grouped under `A_agents` and `B_agents`;
`empty_locations` lists visible empty cells; `model_controlled_agents` lists
visible controlled identities, including the observer, with A/B type and ID.
`rendered_map` gives the same local cells with absolute row/column labels that
wrap around the board (`A`/`B` ordinary, `A#id`/`B#id` controlled, `.` empty).
Distant locations remain hidden. Model-facing types are A/B; saved trajectories
retain their numeric internal types.

Set `tolerance` and `vacancy_fraction` as numbers between zero and one (for example,
`0.75` and `0.25`). Vacancy counts round `board_size ** 2 * vacancy_fraction` to the
nearest integer, with ties to even; population validation still requires equal
type counts. Satisfaction compares the same-type neighbor share directly with
the tolerance, without an epsilon.

Schelling scores final ordinary-to-ordinary edge homophily, excluding the selected
controlled identities in both worlds:

- Integration: `400 * (reference_homophily - controlled_homophily)`.
- Segregation: `400 * (controlled_homophily - reference_homophily)`.

Fractions are from zero to one. Scores are signed and unclamped: an improvement
from 100% to 40% homophily earns 240 integration points.

Civil Violence retains binary, epsilon-free Cascade activation, one-cell
movement for ordinary agents, citizen settlement before police observation,
atomic movement/arrest reservations, and off-grid custody on a
single-occupancy torus. Released citizens start the round inactive and choose
activity and movement on their citizen turn. Model-controlled citizens and
police may move to any unclaimed phase-start vacancy, even outside their local
view. Police can arrest only active citizens adjacent to their chosen destination
when moving, or adjacent to their current position when staying.
Every case replaces exactly 16 existing citizens or police. Either role can seek
increased or decreased participation.

Participation is the active-or-jailed fraction of scored ordinary citizens.
Selected controlled identities are excluded from both worlds. Each world stops
independently at its first completed step with at least 95% participation, or
at the configured `max_steps`; revolution is checked on that last step before
classifying the horizon.

```text
direction = +1 for increase, -1 for decrease
score = direction * (100 * (controlled_participation - reference_participation)
                   + 100 * (controlled_revolution - reference_revolution))
```

Revolution indicators are zero or one. Time to revolution is saved without a
timing reward. Arrest alone does not reduce participation. A jailed model citizen
retains its identity, session, memory, and shared-board access, receives an
explicit jailed observation, and ends its communication-only turn with `defer`.
Physical actions retry; ordinary jailed citizens defer automatically.

The Combined Benchmark Score sums every case score, including negative values,
without averaging or additional weights. Compare totals only for the same suite
and scoring version.
