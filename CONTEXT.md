# MAM-Bench

MAM-Bench evaluates whether locally acting model agents can steer emergent
outcomes in a fixed agent-based system. The project first characterizes the
system's ordinary behavior before defining model-agent evaluations.

## Language

**Benchmark Simulation**:
A versioned MAM-Bench evaluation domain that owns its reference conditions,
ordinary dynamics, model-controlled intervention, and scoring semantics.
Schelling and Civil Violence are separate Benchmark Simulations.
_Avoid_: environment, benchmark case, model

**Model Interaction Protocol**:
The Benchmark Simulation's rules for what model agents observe, which tools they
can use, how they coordinate, and how their outputs become simulation actions.
_Avoid_: prompt, agent loop

**Model Runtime**:
The provider and model execution capability bound to a Benchmark Simulation for
one benchmark run. It executes model requests but does not define the Model
Interaction Protocol.
_Avoid_: model, provider, controller

**Agent Session Runtime**:
The model-agnostic runtime for one trial's isolated persistent agent sessions
and shared communication, leaving each Model Interaction Protocol to its
Benchmark Simulation. Its lifetime ends permanently when the trial fails.
_Avoid_: simulation runtime, agent framework, controller

**Primary Score**:
A Benchmark Simulation's designated higher-is-better topline measure for
comparing models within that simulation. Primary Score magnitudes are not
comparable or aggregated across Benchmark Simulations.
_Avoid_: overall score, universal score, composite score

**Run Evidence**:
The historical persisted model-event and audit bundle for one scored run. It
belongs to the retired evidence contract, not the revised evaluation design.
_Avoid_: artifact, diagnostics, logs

**Evaluation Artifacts**:
The saved scientific results of one evaluation, including the full ordinary and
model-controlled trajectories. They preserve simulation outcomes without the
retired model-event audit bundle.
_Avoid_: Run Evidence, packaged reference fixture

**Diagnostic Artifact**:
Optional retained material that helps inspect a run but is not required to
calculate its Primary Score.
_Avoid_: evidence, run record

**Scientific Analysis**:
Optional derivation from simulation results or a Reference Dataset used to
characterize or improve a benchmark. Scientific Analysis is downstream of
execution and does not change a Primary Score.
_Avoid_: report, dashboard, score

**Schelling Reference Profile**:
A versioned set of Schelling dynamics and parameter coordinates used to generate
a Reference Dataset. It is one explicit model, not every implementation commonly
called a Schelling model.
_Avoid_: canonical Schelling model, original Schelling model

**Staged Round**:
One Reference Profile transition in which agents evaluate satisfaction from one
frozen state, unhappy agents reserve destinations sequentially, and all reserved
moves are then applied together.
_Avoid_: simultaneous update, sequential update

**Reference Dataset**:
Versioned, validated ordinary-behavior data that a Benchmark Simulation uses to
define or evaluate benchmark conditions. Schelling Reference Dataset v2 is one
Reference Dataset.
_Avoid_: baseline dataset, comparison dataset, generated artifacts

**Reference Landscape**:
The outcome distribution produced by ordinary rule-based agents across the
Schelling Reference Profile's parameter space and random seeds. It is used to
locate distinct behavioral regimes before model agents are introduced.
_Avoid_: comparison dataset, baseline dataset

**Reference Trajectory**:
The complete ordered sequence of cell-type grids and agent locations for one
ordinary Schelling run, including its initialized and terminal states and stable
agent types. A Model Evaluation retains and saves its ordinary Counterfactual
Reference's full trajectory.
_Avoid_: run data, history

**Evaluation Reference Fixture**:
The historical packaged projection of a Counterfactual Reference, including its
initial state, terminal state, and comparison values. The revised evaluation
computes its Counterfactual Reference directly instead.
_Avoid_: full dataset, replay log

**Agent Trace**:
The ordered location sequence of one stable agent identity within a Reference
Trajectory. Agent type is static metadata rather than repeated in every state.
_Avoid_: second grid, occupant grid

**Landscape Cell**:
One exact board-size, tolerance, and vacancy coordinate in the Reference
Landscape. Each Landscape Cell contains repeated seeded runs rather than a
single outcome.
_Avoid_: parameter point, configuration

**Landscape Seed**:
A random replicate used to estimate ordinary-agent outcomes in every Landscape
Cell. At a fixed vacancy level, the same Landscape Seed produces a paired
initial grid across tolerance values.
_Avoid_: evaluation seed, trial

**Final Satisfaction**:
The fraction of occupied agents satisfying their exact tolerance rule in a run's
final retained state. It remains defined for equilibrium, blocked, and
horizon-exhausted runs.
_Avoid_: equilibrium satisfaction, similarity

**Satisfaction Manifold**:
The surface over fractional preference and vacancy percentage whose height is
mean Final Satisfaction across Landscape Seeds.
_Avoid_: equilibrium manifold, phase diagram

**Representative Region**:
A bounded part of the Reference Landscape representing a distinct behavioral
regime. Representative Regions remain an analysis concept but are not the unit
selected for the first test panel.
_Avoid_: scenario, case, sample

**Test Spot**:
One exact Landscape Cell selected from the Satisfaction Manifold for held-out
evaluation. The first panel contains three Test Spots rather than broader
Representative Regions.
_Avoid_: representative region, scenario

**Evaluation Seed**:
A held-out random replicate used only after Representative Regions have been
selected. Evaluation Seeds do not contribute to defining the Reference
Landscape's regions.
_Avoid_: landscape seed, trial

**Schelling Influence Profile**:
The single, versioned Model Evaluation contract that layers persistent, locally
embodied ModelControlledAgents onto the Schelling Reference Profile.
_Avoid_: agent variant, LLM Schelling model

**Ordinary Agent**:
A rule-controlled typed board occupant whose satisfaction is evaluated and whose
moves follow the Schelling mechanics for the active evaluation profile.
_Avoid_: algorithmic agent, regular agent

**ModelControlledAgent**:
A model-controlled participant whose observations and permitted actions are
defined by its Benchmark Simulation. In Schelling, it replaces an Ordinary
Agent, retains an exterior A/B type, and has no satisfaction or tolerance of its
own.
_Avoid_: Influence Actor, InfluenceAgent, controlled ordinary agent

**Ordinary Edge Homophily**:
The fraction of undirected toroidal Moore-neighbor edges between scored Ordinary
Agents that join the same type. ModelControlledAgents, and the corresponding
would-be actor identities in a counterfactual reference run, are excluded from
the edge set.
_Avoid_: segregation index, satisfaction

**Message Board**:
The append-only ordered communication surface shared by ModelControlledAgents in
one trial. Agent posts are unverified free text, simulation announcements are
distinguished from agent posts, and each agent has an independent unread
position.
_Avoid_: Public Document, shared notebook, controller

**Actor Notebook**:
One ModelControlledAgent's private persistent notes within a single trial,
isolated from other agents and reset before the next trial.
_Avoid_: Message Board, shared memory, transcript

**Actor Turn**:
One ModelControlledAgent's opportunity to act in one Staged Round, beginning
with a frozen local observation and ending with an accepted move or a stay.
_Avoid_: wave, request, round

**Turn Tool Allowance**:
The maximum number of successful function-tool calls available to one
ModelControlledAgent during an Actor Turn. Failed calls are handled through
model retries; terminal move and stay outputs end the turn.
_Avoid_: attempt count, token budget, request count

**Policy Rejection**:
Historical terminology for the removed custom turn policy. Current invalid
tools and actions use native model retry feedback, with exhausted retries
ending the turn as stay. Infrastructure failures remain distinct.
_Avoid_: provider error, failed pair, invalid score

**Reservation Record**:
An authoritative Message Board entry written when a ModelControlledAgent
atomically claims a beginning-of-round vacancy.
_Avoid_: actor post, move result, proposal

**Pair Infrastructure Failure**:
A provider, network, timeout, memory, compaction, or artifact-write failure
that ends one simulation/model pair and leaves it unscored without stopping
independent pairs. A new attempt starts a fresh trial rather than resuming the
failed one.
_Avoid_: policy rejection, stay, zero score

**Counterfactual Reference**:
The all-Ordinary-Agent run computed before a Model Evaluation from its same
initial condition. Its values remain in memory and supply the ordinary outcome
comparator.
_Avoid_: landscape average, reference landscape cell

**Directional Lift**:
The signed difference between model-run and Counterfactual Reference final
Ordinary Edge Homophily, oriented so positive values always indicate movement
toward the assigned integration or segregation objective.
_Avoid_: improvement, normalized score

**Model Evaluation**:
A later experiment that introduces ModelControlledAgents into frozen conditions
selected from the Reference Landscape. It is downstream of reference generation
and Test Spot selection.
_Avoid_: benchmark run, model test
