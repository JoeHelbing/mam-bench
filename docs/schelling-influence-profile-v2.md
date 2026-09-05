# Schelling Influence Profile v2

## Purpose and scope

This document is the normative contract for `schelling-influence-v2`. The
evaluation asks whether 16 persistent, locally embodied model-controlled
Influence Actors can steer a fixed Schelling system toward an assigned
integration or segregation objective. It does not claim that the board predicts
a real society.

The YAML simulation identifier remains `schelling-influence-pilot-v1` for
operator compatibility. Result metadata identifies the scientific protocol as
`schelling-influence-v2`; results from earlier protocol versions are not directly
comparable.

## Fixed evaluation condition

The current evaluation uses the Schelling Reference Profile v2 condition with:

- a 20x20 toroidal board;
- exact tolerance `3/4`;
- vacancy fraction `1/4`;
- held-out Evaluation Seed `50`; and
- the integration objective.

Landscape Seeds remain `0..49`. The packaged Evaluation Reference Fixture
provides the same-seed initial state, stable identities, terminal
Counterfactual Reference state, termination facts, and comparison metrics. The
model evaluation always executes 30 Staged Rounds even though this reference
reached equilibrium after 20 transitions.

## Population, authority, and score

Every board contains 300 occupied cells. Sixteen fixed identities become
Influence Actors: type A IDs `0..7` and type B IDs `150..157`. The other 284
identities remain Ordinary Agents.

An Influence Actor retains its immutable exterior A/B type, which Ordinary
Agents observe when evaluating their neighborhoods. It has no tolerance or
satisfaction, never enters the unhappy set, and may act only through the
Influence Profile tools. It cannot change type, move another identity, or mutate
the board directly.

Ordinary Edge Homophily is the fraction of undirected toroidal radius-1 Moore
edges between scored Ordinary Agents whose endpoints have the same type. The 16
Influence Actors are removed from the model-run edge set. The same 16 would-be
actor identities are removed from the Counterfactual Reference edge set.

Directional Lift is higher-is-better:

- integration: `reference_final_homophily - model_final_homophily`;
- segregation: `model_final_homophily - reference_final_homophily`.

The final Directional Lift is the Primary Score. Best Directional Lift and the
mean directional lift across the 31 retained states are diagnostics, as are
Ordinary Agent satisfaction, accepted actor moves, policy rejections, model
usage, and latency. One selected model receives one stochastic trial. The
result is one sample, not an estimate or confidence interval.

## Agent Session Runtime boundary

MAM-Bench's generic Agent Session Runtime composes PydanticAI and PydanticAI
Harness. One runtime is created per simulation/model trial. It owns:

- one normalized persistent message history per opaque actor session;
- one in-memory notebook store with isolated actor namespaces that reset between trials;
- notebook injection and context compaction;
- append-only opaque shared communication with per-session cursors;
- bounded rolling request admission;
- normalized model messages, monotonic event recording, and usage accounting.

The runtime does not define Schelling observations, prompts, tools, call policy,
record meaning, movement rules, scoring, or required evidence. Those remain the
Benchmark Simulation's Model Interaction Protocol.

## Actor Turn protocol

Each actor receives one Actor Turn in every round. An Actor Turn is one
persistent PydanticAI run and may end before its call budget is exhausted.

At turn start the actor receives only:

- its stable actor ID, exterior type, and absolute coordinate;
- the round number and 30-round horizon;
- one frozen radius-1 toroidal Moore neighborhood;
- neighboring Ordinary Agent types, neighboring Influence Actor IDs and types,
  and visible vacancies;
- the integration or segregation objective;
- current masked Ordinary Edge Homophily;
- fixed final masked Counterfactual Reference homophily;
- the current count of unreserved beginning-of-round vacancies; and
- the ten-Successful-Call budget.

The actor does not receive the global board, the reference board, satisfaction
labels, dissatisfied-agent locations, reservation locations, or computed move
recommendations. The local neighborhood is not refreshed during the turn.

Every model response must contain exactly one available function-tool or
terminal-output call. Co-emitted or malformed calls are rejected before side
effects. Parallel tool calls are disabled where the provider supports that
setting, but protocol correctness does not depend on provider compliance.

The nonterminal actions are:

- `read_document`: read the oldest unread complete Public Document records;
- `post_message`: append free text to the Public Document; and
- the standard notebook read, write, search, and delete tools.

The terminal actions are `submit_move(row, column)` and `stay`. Either may be
used early and ends the turn. A successful post, move, or stay may carry one
full notebook append, replacement, or deletion at no additional call cost.

Only successfully executed function tools and successfully committed terminal
outputs count as Successful Calls. Rejected attempts do not consume that
budget. After a successful nonterminal call the model receives the remaining
count. When one call remains, only move and stay are available. Two
model-policy retries are permitted across the complete turn; a third Policy
Rejection creates a recorded runtime-authored stay without another model
request. Infrastructure failures are not converted into stays.

## Private memory and compaction

Each actor has one private Actor Notebook for the trial. Its stable identity,
history, cursor, and notebook namespace persist across all 30 rounds and are
isolated from other actors and trials. `MEMORY.md` is automatically injected on
every model request within an approximately 4,000-token budget. Other notebook
files remain available through the Harness tools. The main notebook cannot be
deleted.

The coordinator validates an action before applying its attached notebook
change, then commits the accepted public action under the same lock. Rejected
actions cannot mutate memory. Attached changes use the public Harness
`MemoryToolset` with the actual tool or output `RunContext`, so Harness owns
file validation, editing, bounds, and conflict behavior.

A scored trial contains consistent accepted actions and notebook changes. A
memory-store or later infrastructure failure aborts the pair and discards its
runtime; completed notebook writes are not rolled back for reuse.

The protocol fixes a 260,000-token context window. At 70 percent, Harness uses
the evaluated model for incremental summarization, retains a 40,000-token
verbatim tail, and permits up to 16,000 completion tokens for the summary.
Approved summarization/model failures fall back to a deterministic 40,000-token
sliding window. Compaction messages, receipts, fallback use, requests, tokens,
and available cost are included in normalized evidence. The runtime does not
use provider-native compaction or silently reduce the window for an incompatible
model.

## Public Document

The Public Document is an append-only ordered sequence shared by all 16 actors.
Actor posts preserve arbitrary free text, identify their author, are explicitly
unverified, and are limited to 4,000 characters. Runtime-authored authoritative
records use a distinct record type.

Actors retrieve records explicitly through `read_document`; posts are not
automatically injected. Each actor has an independent chronological cursor. A
read returns only complete records within 40,000 rendered characters and states
whether more unread records remain. An empty successful read still consumes one
Successful Call. The cursor advances at the read's authoritative lock point.

## Rolling concurrency and reservations

Each round derives a deterministic randomized actor-admission permutation from
the fixed semantic coordinates, Evaluation Seed, round number, and an
influence-only NumPy `SeedSequence`/`PCG64` stream. This stream is separate from
ordinary-agent reservation order and tie breaking.

Actors enter a rolling queue up to the configured concurrency. As one Actor
Turn completes, the next may enter; there are no synchronized communication or
movement waves. Completion and coordinator-lock acquisition order determine
Public Document publication and actor reservation priority. That order is
intentionally stochastic and is recorded rather than promised to replay
bit-for-bit.

All movement eligibility uses the frozen beginning-of-round board:

1. A move may request any coordinate that was vacant at round start, whether or
   not it appeared in the local neighborhood.
2. The coordinator atomically accepts the first reservation for an available
   destination and publishes an authoritative Reservation Record.
3. A later request for that destination is a Policy Rejection and may be
   retried within the turn allowance.
4. Actor origins do not become eligible destinations during the same round.
5. After all Actor Turns complete, unhappy Ordinary Agents reserve nearest
   satisfactory vacancies using the Reference Profile's seeded order and
   true-tie stream, excluding accepted actor destinations.
6. Accepted actor and Ordinary Agent moves apply together at settlement.

Ordinary equilibrium, blocked status, all actors staying, or a repeated board
does not end the evaluation early. The run retains initialization plus all 30
settled states.

## Model settings and repeatability

The selected Model Runtime supplies provider/model construction and provenance.
Current agent settings use temperature `1.0`, top-p `0.95`, top-k `20`, medium
reasoning effort, a 32,768-token completion ceiling, configured rolling
concurrency, and the fixed context/compaction contract above. Shared request
settings live on the model so actor and compaction requests inherit the same
provider policy. OpenRouter uses the native `OpenRouterModel` with typed routing
settings that pin the selected provider, require declared parameters, and
disable fallbacks. Generic compatible endpoints use `OpenAIChatModel`.
Compaction overrides the completion ceiling with its fixed 16,000-token limit.

Each Actor Turn receives an advisory deterministic 32-bit model sampling seed
derived from the public master words, objective, Landscape Cell, Evaluation
Seed, round, phase, and actor ID. Providers may not offer deterministic model
execution. Scheduler and model nondeterminism are expected protocol behavior.

## Run Evidence and failure behavior

A successful pair is built in a temporary sibling directory, validated, and
atomically published to a destination that must not already exist. It contains
exactly:

- `run.json`: configuration, versioned protocol and runtime provenance, score,
  diagnostics, and aggregate usage;
- `trajectory.npz`: the 31 typed board/location states and compact per-round
  numeric measures, loaded with `allow_pickle=False`; and
- `events.jsonl`: the canonical normalized, monotonically sequenced event
  stream.

The event stream records normalized model-visible messages once and references
them by stable IDs. It also records admission, turn timing, document reads and
publication, reservations, stays, movement settlement, memory changes,
notebook snapshots, Policy Rejections, compaction, and usage. Provider client
objects, raw HTTP payloads, headers, credentials, and secret-shaped values are
not evidence.

A Pair Infrastructure Failure, including provider, network, timeout, memory,
unsupported-context, compaction, or evidence-publication failure, aborts that
simulation/model pair. The runtime stops admitting turns, cancels and awaits
remaining actors, and cannot be reused or resumed. A new attempt starts a fresh
trial; it never reads the failed runtime's histories or notebooks.
The pair remains unscored and publishes exactly `failure.json` plus redacted
`failed-events.jsonl`. Other independent pairs continue. After the matrix, the
benchmark writes `topline.json` from successful pairs only and exits nonzero if
any pair failed. A partial topline is therefore evidence of completed successes,
not a claim that the whole requested matrix passed.
