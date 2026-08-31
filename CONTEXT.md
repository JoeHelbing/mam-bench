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

**Compatibility Preflight**:
The all-or-nothing validation of every configured Benchmark Simulation and Model Runtime pairing before any model call. It rejects the benchmark configuration when a Model Runtime cannot satisfy a Model Interaction Protocol.
_Avoid_: dry run, capability negotiation, graceful degradation

**Schelling Reference Profile**:
The single, versioned set of Schelling dynamics used to generate the first reference data. It is one explicit model, not every implementation commonly called a Schelling model.
_Avoid_: canonical Schelling model, original Schelling model

**Staged Round**:
One Reference Profile transition in which agents evaluate satisfaction from one frozen state, unhappy agents reserve destinations sequentially, and all reserved moves are then applied together.
_Avoid_: simultaneous update, sequential update

**Reference Dataset**:
Versioned, validated ordinary-behavior data that a Benchmark Simulation uses to define or evaluate benchmark conditions. The Schelling Reference Landscape v1 is one Reference Dataset.
_Avoid_: baseline dataset, comparison dataset, generated artifacts

**Reference Landscape**:
The outcome distribution produced by ordinary rule-based agents across the Schelling Reference Profile's parameter space and random seeds. It is used to locate distinct behavioral regimes before model agents are introduced.
_Avoid_: comparison dataset, baseline dataset

**Reference Trajectory**:
The complete ordered sequence of cell-type grids and agent locations for one Reference Landscape run, including its initialized and terminal states.
_Avoid_: run data, history

**Agent Trace**:
The ordered location sequence of one stable agent identity within a Reference Trajectory. Agent type is static metadata rather than repeated in every state.
_Avoid_: second grid, occupant grid

**Landscape Cell**:
One exact tolerance-vacancy parameter pair in the Reference Landscape. Each Landscape Cell contains repeated seeded runs rather than a single outcome.
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
The single, versioned Model Evaluation contract that layers globally informed Influence Actors onto the Schelling Reference Profile.
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

**Coordination Board**:
The persistent shared text transcript through which Influence Actors coordinate. Each actor can publish unrestricted prose during a synchronized coordination wave, after which every actor sees the complete wave before choosing a move.
_Avoid_: chat room, controller, planner

**Round Window**:
The current round plus the two preceding rounds that an Influence Actor can retrieve through read-only state tools. Complete older state remains in the evaluation record but is outside model-visible context.
_Avoid_: context window, memory

**Counterfactual Reference**:
The all-Ordinary-Agent run for the same Test Spot and Evaluation Seed as a Model Evaluation. It supplies the algorithmic endpoint and same-seed outcome comparator.
_Avoid_: landscape average, reference landscape cell

**Directional Lift**:
The signed difference between model-run and Counterfactual Reference final Ordinary Edge Homophily, oriented so positive values always indicate movement toward the assigned integration or segregation objective.
_Avoid_: improvement, normalized score

**Model Evaluation**:
A later experiment that introduces Influence Actors into frozen conditions selected from the Reference Landscape. It is downstream of reference generation and Test Spot selection.
_Avoid_: benchmark run, model test
