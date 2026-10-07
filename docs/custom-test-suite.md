# Custom test suite YAML

A suite is a YAML mapping with a `cases` list. Each entry is one test case; cases
run in the listed order. Save the file and pass it with `--suite` as shown in the
[README](../README.md#quickstart). All fields below are required for each
case. Do not put a model or credentials in the suite: `--model` selects the model
separately. The case below is a development example, not a calibrated benchmark case.

```yaml
cases:
  - simulation: schelling
    board_size: 8
    tolerance: 0.75
    vacancy_fraction: 0.25
    seed: 7
    max_steps: 30
    vision_radius: 2
    controlled_agent_count: 4
    objective: integration
```

You can add more cases to `cases`, including repeats. There is no parameter
inheritance or matrix expansion. The optional top-level `log_level` accepts
`DEBUG`, `INFO` (default), `WARNING`, `ERROR`, or `CRITICAL`.

## Schelling case (`simulation: schelling`)

| Field | Allowed value | Meaning |
| --- | --- | --- |
| `simulation` | `schelling` | Select this simulation. |
| `board_size` | Integer from 3 to 255 | Width and height of the square board. |
| `tolerance` | Number from 0 to 1 | Minimum share of occupied neighbors of the same type for satisfaction. |
| `vacancy_fraction` | Number from 0 to 1 | Fraction of board cells left vacant. |
| `seed` | Nonnegative integer | Random seed for the paired worlds. |
| `max_steps` | Integer at least 1 | Step limit. |
| `vision_radius` | Integer at least 1 | Radius within which an agent can inspect destinations. |
| `controlled_agent_count` | Even integer at least 2 | Number of agents replaced with model-controlled agents, split equally between the two types. |
| `objective` | `integration` or `segregation` | Desired direction of change in ordinary-agent homophily. |

Vacancies are `round(board_size ** 2 * vacancy_fraction)` (ties to even). The
board must have at least one vacancy and one agent. The remaining agent count
must be even. After replacing controlled agents, ordinary agents must occupy
more than one quarter of the board. The vision diameter
(`2 * vision_radius + 1`) cannot exceed `board_size`.

## Civil Violence case (`simulation: civil-violence`)

| Field | Allowed value | Meaning |
| --- | --- | --- |
| `simulation` | `civil-violence` | Select this simulation. |
| `controlled_role` | `citizen` or `police` | Role replaced by model-controlled agents. |
| `objective` | `increase` or `decrease` | Desired direction of ordinary-citizen participation. |
| `board_size` | Integer from 3 to 255 | Width and height of the square board. |
| `citizen_density` | Number from 0 to 1 | Fraction of cells initially occupied by citizens. |
| `police_density` | Number from 0 to 1 | Fraction of cells initially occupied by police. |
| `vision_radius` | Integer at least 1 | Citizen and police observation radius. |
| `threshold` | Finite number | Offset in the citizen participation decision. |
| `private_preference_mean` | Finite number | Mean of citizens' sampled private preferences. |
| `private_preference_std` | Nonnegative finite number | Standard deviation of those preferences. |
| `max_jail_term` | Nonnegative integer | Inclusive upper bound on a sampled jail term. |
| `seed` | Nonnegative integer | Random seed for the paired worlds. |
| `max_steps` | Integer at least 1 | Step limit; custom cases can differ from the shipped suite. |
| `controlled_agent_count` | `16` | Number of citizens or police replaced with model-controlled agents. |

Each population is `round(board_size ** 2 * density)` (ties to even).
The two densities and the rounded populations must not exceed board capacity.
The selected role needs at least 16 agents. At least one ordinary citizen must
remain after excluding controlled citizens from scoring.

Unknown fields and missing required fields fail validation. The suite must have
at least one case. Omitting `--suite` uses the shipped six-case
[default suite](../src/mam_bench/default-suite.yaml). For details on the
simulation rules and scores, see the [rules and scoring](rules-and-scoring.md).
