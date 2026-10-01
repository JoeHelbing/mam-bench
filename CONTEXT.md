# MAM-Bench

MAM-Bench evaluates whether locally acting model agents can steer emergent
outcomes in a fixed agent-based system. The project first characterizes the
system's ordinary behavior before defining model-agent evaluations.

## Language

**Model**:
The single language model being evaluated in a Benchmark Suite invocation.
Model-controlled agents use that selected model; Schelling and Civil Violence
are Benchmark Simulations, not Models in this vocabulary.
_Avoid_: simulation, individual agent

**Simulation Agent**:
One individual participant in a simulation world, such as a Schelling occupant
or a Civil Violence citizen or police officer. It has a stable identity and
chooses actions through ordinary rules or model control.
_Avoid_: model interface, language model

**Model Interface**:
The shared definition of how simulation agents consult the selected Model,
including instructions and available tools for their simulation and role.
It is distinct from each participating Simulation Agent and its Agent Session.
_Avoid_: citizen, agent session

**Agent Session**:
One model-controlled Simulation Agent's private conversation history and memory
throughout one world run. Sessions remain isolated by identity while sharing
the applicable Model Interface and Message Board.
_Avoid_: model interface, benchmark run

**Benchmark Simulation**:
A versioned MAM-Bench evaluation domain that owns its reference conditions,
ordinary dynamics, model-controlled intervention, and scoring semantics.
Schelling and Civil Violence are separate Benchmark Simulations, shortened to
Simulations in conversation.
_Avoid_: environment, benchmark case, model

**Test Case**:
One configured evaluation of a Simulation with a specific parameter set,
objective, seed, and controlled role where applicable, executed once for the
selected Model. It produces one score contribution to the Benchmark Run.
_Avoid_: simulation type, benchmark run

**Benchmark Run**:
One start-to-finish execution of the Benchmark Suite against exactly one
explicitly selected Model. A successful run produces one Combined Benchmark
Score; a failed run retains partial artifacts without a final score.
_Avoid_: individual test case, multi-model campaign

**Model Interaction Protocol**:
The Benchmark Simulation's rules for what model agents observe, which tools they
can use, how they coordinate, and how their outputs become simulation actions.
_Avoid_: prompt, agent loop

**Civil Violence Activity**:
The active or inactive behavioral state of a free citizen in the Civil Violence
Benchmark Simulation. There is no intermediate opposed state; imprisonment
remains a separate restriction on participation.
_Avoid_: support/oppose/active spectrum

**Tactical Participation**:
The Civil Violence intervention in which a model-controlled citizen chooses
movement and activity, or a model-controlled police officer chooses movement
and an eligible arrest target or no arrest, within the simulation's role rules.
_Avoid_: movement-only control

**Civil Violence Cycle**:
One citizen participation phase followed by one police intervention phase,
after which ordinary-citizen participation and revolution are measured.
_Avoid_: globally simultaneous round

**Arrest Reservation**:
An exclusive claim by a police officer on one eligible active citizen for
arrest at the end of the police phase. A reservation is not an immediate arrest.
_Avoid_: completed arrest

**Civil Violence Objective**:
The assigned direction of desired change in citizen participation, either
increase or decrease, independent of the model-controlled role. Citizen control
and police control can each be evaluated under either objective.
_Avoid_: citizen goal, police goal

**Civil Violence Participation**:
The fraction of scored ordinary citizens who are active or jailed, with jailed
citizens included in both numerator and denominator. This is the outcome measure
for both objectives under either controlled role; controlled identities are
excluded in both paired worlds.
_Avoid_: free active share, cumulative ever-active share

**Revolution**:
A Civil Violence event in which at least 95% of ordinary citizens are active or
jailed at any one turn; persistence across turns is not required. Jailed citizens
count in both numerator and denominator; model-controlled identities are
excluded in both the model run and its ordinary reference. The threshold is
checked after each complete Civil Violence Cycle, once citizen and security
actions have resolved. Its first occurrence ends the run, and that completed
cycle is recorded as the time to revolution.
_Avoid_: sustained activity, regime collapse

**Primary Score**:
A Benchmark Simulation's designated higher-is-better topline measure for
comparing models within that simulation. Primary Score magnitudes are not
directly comparable across Benchmark Simulations.
_Avoid_: overall score, universal score, composite score

**Combined Benchmark Score**:
The sum of signed per-test scores across all evaluated simulation, objective,
condition, and seed variations, with no averaging or additional aggregation
weights. Gains add and deterioration subtracts; totals are comparable only for
the same test suite and scoring version.
_Avoid_: absolute goal achievement, percentage improvement over baseline

**Benchmark Suite**:
The versioned collection of Test Cases executed by a Benchmark Run for one
explicitly selected Model. Simulation defaults are selected as a suite; their
final selection is deferred.
Model selection is always explicit.
_Avoid_: model preset, single simulation run

**Evaluation Artifacts**:
The saved results and inspectable interaction records of an evaluation,
including ordinary and model-controlled trajectories, scores, agent model
messages, tool calls, and shared Message Board communication. Records are
retained as execution progresses, including partial work before failure.
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
frozen state. Selected replacement identities reserve first, followed by the
remaining ordinary identities; each group follows its seeded Turn Priority.
All reserved moves are then applied together.
_Avoid_: immediate sequential movement, response-order settlement

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
The fraction of occupied agents satisfying their tolerance rule in a run's
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
evaluation. Test Spot and seed selection are deferred.
_Avoid_: representative region, scenario

**Evaluation Seed**:
A held-out random replicate used only after Representative Regions have been
selected. Evaluation Seeds do not contribute to defining the Reference
Landscape's regions.
_Avoid_: landscape seed, trial

**Schelling Influence Profile**:
The single, versioned Model Evaluation contract that layers persistent, locally
embodied model-controlled agents onto the Schelling Reference Profile.
_Avoid_: agent variant, LLM Schelling model

**Ordinary Agent**:
A Simulation Agent that selects actions through its simulation's ordinary
rules. These are Schelling movement decisions, Civil Violence citizen
participation, or Civil Violence police intervention.
_Avoid_: algorithmic agent, regular agent

**Model-controlled Agent**:
A model-controlled participant whose observations and permitted actions are
defined by its Benchmark Simulation. In Schelling, it replaces an Ordinary
Agent, retains an exterior A/B type, and has no satisfaction or tolerance of its
own. Model-controlled agents are outside the population used for simulation
outcome statistics; corresponding identities are also excluded from reference
statistics.
_Avoid_: Influence Actor, InfluenceAgent, controlled ordinary agent

**Ordinary Edge Homophily**:
The fraction of undirected toroidal Moore-neighbor edges between scored Ordinary
Agents that join the same type. Model-controlled agents, and the corresponding
selected identities in a counterfactual reference run, are excluded from
the edge set.
_Avoid_: segregation index, satisfaction

**Message Board**:
The append-only ordered communication surface shared by model-controlled agents in
one controlled world. Agent posts are unverified free text, simulation announcements are
distinguished from agent posts, and each agent has an independent unread
position.
_Avoid_: Public Document, shared notebook, controller

**Private Notebook**:
One model-controlled agent's private persistent notes within one world,
isolated from other agents. Each test case creates fresh notebooks.
_Avoid_: Message Board, shared memory, transcript

**Agent Turn**:
One model-controlled agent's opportunity to act in its simulation's current
step or role phase. It receives a frozen observation, can communicate and use
private memory, and ends with a legal role-specific action or fallback.
_Avoid_: wave, request, round

**Turn Priority**:
The seeded order in which a group's action claims are accepted. Corresponding
identities have the same priority in paired worlds, regardless of decision policy
or response speed. Priority is distinct from concurrent decision-making.
_Avoid_: response order, wall-clock order

**Paired Random Slot**:
The randomness assigned to one identity and decision purpose in one step and
role phase. Corresponding slots match across paired worlds; an unused slot does
not shift other slots when policies, custody, or termination differ.
_Avoid_: shared mutable random stream, model sampling seed

**Turn Tool Allowance**:
The maximum number of successful function-tool calls available to one
model-controlled agent during an Agent Turn. Failed calls are handled through
model retries; an accepted terminal action ends the turn.
_Avoid_: attempt count, token budget, request count

**Reservation Record**:
An authoritative Message Board entry written when a model-controlled agent
atomically claims a beginning-of-round vacancy.
_Avoid_: actor post, move result, proposal

**Execution Failure**:
A provider, network, timeout, memory, compaction, persistence, or simulation
failure that stops the Benchmark Run. Completed results and partial artifacts
remain available, but no Combined Benchmark Score is published.
_Avoid_: policy rejection, stay, zero score

**Counterfactual Reference**:
The all-Ordinary-Agent run computed before a Model Evaluation from its same
initial condition, selected-identity priority, and Paired Random Slots. Its values
remain in memory and supply the ordinary outcome comparator.
_Avoid_: landscape average, reference landscape cell

**Directional Lift**:
The signed difference between model-run and Counterfactual Reference final
Ordinary Edge Homophily, oriented so positive values always indicate movement
toward the assigned integration or segregation objective.
_Avoid_: improvement, normalized score

**Model Evaluation**:
A paired comparison of a controlled world against its Counterfactual Reference
for one Test Case. Benchmark calibration and final real-model evaluations are
separate from implementation checks using development conditions.
_Avoid_: benchmark run, model test
