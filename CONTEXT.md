# MAM-Bench

MAM-Bench evaluates whether locally acting model agents can steer emergent outcomes in a fixed agent-based system. The project first characterizes the system's ordinary behavior before defining model-agent evaluations.

## Language

**Benchmark Simulation**:
A versioned MAM-Bench evaluation domain that owns its reference conditions, ordinary dynamics, model-controlled intervention, scoring semantics, validation, and required evidence. Schelling and Civil Violence are separate Benchmark Simulations.
_Avoid_: environment, benchmark case, model

**Model Interaction Protocol**:
The Benchmark Simulation's rules for what model agents observe, which tools they can use, how they coordinate, and how their outputs become simulation actions.
_Avoid_: prompt, agent loop

**Model Runtime**:
The provider and model execution capability bound to a Benchmark Simulation for one benchmark run. It returns auditable model responses but does not define the Model Interaction Protocol.
_Avoid_: model, provider, controller

**Agent Session Runtime**:
The model-agnostic runtime for one trial's isolated persistent agent sessions and shared communication, leaving each Model Interaction Protocol to its Benchmark Simulation. Its lifetime ends permanently when the trial fails.
_Avoid_: simulation runtime, agent framework, controller

**Primary Score**:
A Benchmark Simulation's designated higher-is-better topline measure for comparing models within that simulation. Primary Score magnitudes are not comparable or aggregated across Benchmark Simulations.
_Avoid_: overall score, universal score, composite score

**Run Evidence**:
The records a Benchmark Simulation requires to validate and audit one scored run. A run without its complete required Run Evidence is unscored.
_Avoid_: artifact, diagnostics, logs

**Diagnostic Artifact**:
Optional retained material that helps inspect a run but is not required to validate its Primary Score. Benchmark configuration may suppress Diagnostic Artifacts without suppressing Run Evidence.
_Avoid_: evidence, run record

**Scientific Analysis**:
Optional machine-readable derivation from a Reference Dataset or validated Run Evidence used to characterize, audit, or improve a benchmark. Scientific Analysis is downstream of execution: it never changes a Primary Score, completes Run Evidence, or enters the runtime dependency direction.
_Avoid_: report, dashboard, score

**Schelling Reference Profile**:
A versioned set of Schelling dynamics and parameter coordinates used to generate a Reference Dataset. It is one explicit model, not every implementation commonly called a Schelling model.
_Avoid_: canonical Schelling model, original Schelling model

**Staged Round**:
One Reference Profile transition in which agents evaluate satisfaction from one frozen state, unhappy agents reserve destinations sequentially, and all reserved moves are then applied together.
_Avoid_: simultaneous update, sequential update

**Reference Dataset**:
Versioned, validated ordinary-behavior data that a Benchmark Simulation uses to define or evaluate benchmark conditions. Schelling Reference Dataset v2 is one Reference Dataset.
_Avoid_: baseline dataset, comparison dataset, generated artifacts

**Reference Landscape**:
The outcome distribution produced by ordinary rule-based agents across the Schelling Reference Profile's parameter space and random seeds. It is used to locate distinct behavioral regimes before model agents are introduced.
_Avoid_: comparison dataset, baseline dataset

**Reference Trajectory**:
The complete ordered sequence of cell-type grids and agent locations for one Reference Landscape run, including its initialized and terminal states. Full Schelling Reference Trajectories are developer analysis material and are not packaged.
_Avoid_: run data, history

**Evaluation Reference Fixture**:
The compact packaged projection of one held-out Counterfactual Reference needed by a Benchmark Simulation: fixed parameters, initial identities and state, terminal state, termination facts, and comparison metrics. It excludes unused intermediate Reference Trajectory states.
_Avoid_: full dataset, replay log

**Agent Trace**:
The ordered location sequence of one stable agent identity within a Reference Trajectory. Agent type is static metadata rather than repeated in every state.
_Avoid_: second grid, occupant grid

**Landscape Cell**:
One exact board-size, tolerance, and vacancy coordinate in the Reference Landscape. Each Landscape Cell contains repeated seeded runs rather than a single outcome.
_Avoid_: parameter point, configuration

**Landscape Seed**:
A random replicate used to estimate ordinary-agent outcomes in every Landscape Cell. At a fixed vacancy level, the same Landscape Seed produces a paired initial grid across tolerance values.
_Avoid_: evaluation seed, trial

**Final Satisfaction**:
The fraction of occupied agents satisfying their exact tolerance rule in a run's final retained state. It remains defined for equilibrium, blocked, and horizon-exhausted runs.
_Avoid_: equilibrium satisfaction, similarity

**Satisfaction Manifold**:
The surface over fractional preference and vacancy percentage whose height is mean Final Satisfaction across Landscape Seeds.
_Avoid_: equilibrium manifold, phase diagram

**Representative Region**:
A bounded part of the Reference Landscape representing a distinct behavioral regime. Representative Regions remain an analysis concept but are not the unit selected for the first test panel.
_Avoid_: scenario, case, sample

**Test Spot**:
One exact Landscape Cell selected from the Satisfaction Manifold for held-out evaluation. The first panel contains three Test Spots rather than broader Representative Regions.
_Avoid_: representative region, scenario

**Evaluation Seed**:
A held-out random replicate used only after Representative Regions have been selected. Evaluation Seeds do not contribute to defining the Reference Landscape's regions.
_Avoid_: landscape seed, trial

**Schelling Influence Profile**:
The single, versioned Model Evaluation contract that layers persistent, locally embodied Influence Actors onto the Schelling Reference Profile.
_Avoid_: agent variant, LLM Schelling model

**Ordinary Agent**:
A rule-controlled typed board occupant whose satisfaction is evaluated and whose moves follow the Schelling mechanics for the active evaluation profile.
_Avoid_: algorithmic agent, regular agent

**Influence Actor**:
A model-controlled typed board occupant that replaces an Ordinary Agent while retaining an exterior A/B type visible to neighbors. It has no satisfaction or tolerance of its own and is excluded from satisfaction scoring.
_Avoid_: AI agent, controlled ordinary agent, model agent

**Ordinary Edge Homophily**:
The fraction of undirected toroidal Moore-neighbor edges between scored Ordinary Agents that join the same type. Influence Actors, and the corresponding would-be actor identities in a counterfactual reference run, are excluded from the edge set.
_Avoid_: segregation index, satisfaction

**Public Document**:
The append-only ordered communication surface shared by Influence Actors. Actor posts are unverified free text; runtime-authored facts use a distinct authoritative record type; each actor reads through an independent cursor.
_Avoid_: chat room, coordination board, controller

**Actor Notebook**:
One Influence Actor's private persistent Harness memory within a single trial, isolated from other actors and reset before the next trial.
_Avoid_: public document, shared memory, transcript

**Actor Turn**:
One persistent model run for one Influence Actor in one Staged Round, beginning with a frozen local observation and ending with an accepted move, stay, or runtime-authored forced stay.
_Avoid_: wave, request, round

**Successful Call**:
A function tool that executed successfully or a terminal output that committed successfully during an Actor Turn. Rejected attempts and model requests are not Successful Calls.
_Avoid_: request, attempt, tool emission

**Policy Rejection**:
A model response or proposed action rejected by the active Model Interaction Protocol before its public, reservation, or memory side effects commit. It is distinct from infrastructure failure.
_Avoid_: provider error, failed pair, invalid score

**Reservation Record**:
An authoritative Public Document entry written when an Influence Actor atomically claims a beginning-of-round vacancy.
_Avoid_: actor post, move result, proposal

**Pair Infrastructure Failure**:
A provider, network, timeout, memory, unsupported-context, compaction, or evidence-publication failure that ends one simulation/model pair, leaves it unscored, and preserves redacted failure evidence without stopping independent pairs. A new attempt starts a fresh trial rather than resuming the failed one.
_Avoid_: policy rejection, stay, zero score

**Counterfactual Reference**:
The all-Ordinary-Agent run for the same Test Spot and Evaluation Seed as a Model Evaluation. It supplies the algorithmic endpoint and same-seed outcome comparator.
_Avoid_: landscape average, reference landscape cell

**Directional Lift**:
The signed difference between model-run and Counterfactual Reference final Ordinary Edge Homophily, oriented so positive values always indicate movement toward the assigned integration or segregation objective.
_Avoid_: improvement, normalized score

**Model Evaluation**:
A later experiment that introduces Influence Actors into frozen conditions selected from the Reference Landscape. It is downstream of reference generation and Test Spot selection.
_Avoid_: benchmark run, model test
