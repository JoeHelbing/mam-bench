# Schelling Influence v2 implementation plan

## Goal

Replace the current two-wave global-inspection protocol with a 30-round embodied-actor protocol built directly on PydanticAI and PydanticAI Harness.

Keep:

- `simulation_id = "schelling-influence-pilot-v1"`;
- the fixed 20x20, tolerance `3/4`, vacancy `1/4`, Evaluation Seed 50 case;
- Reference Dataset v2 and deterministic ordinary-agent mechanics;
- `BenchmarkSimulation` and `ModelRuntime` as the generic seams;
- OpenRouter and arbitrary OpenAI-compatible providers;
- `uv run main.py CONFIG.yaml`;
- no-overwrite and credential-safety guarantees.

Advance `simulation_version` to `schelling-influence-v2`.

Normative spec: <https://github.com/JoeHelbing/mam-bench/issues/1>

Model-agnostic agent mechanisms belong at the `mam_bench` package level. A narrow Agent Session Runtime owns persistent histories, Harness memory, compaction, shared communication storage, rolling admission, normalized model evidence, and usage aggregation. Schelling owns its tools, outputs, prompts, injected information, call policy, record meaning, movement, scoring, and required Run Evidence.

## Frozen protocol

### Actor turns

- The 16 persistent Influence Actors remain IDs `0..7` and `150..157`.
- One reusable PydanticAI `Agent` definition serves all actors. Histories, document cursors, call counters, and memory namespaces remain actor-specific.
- The influence evaluation runs exactly 30 rounds and retains 31 board states. The deterministic Seed 50 reference remains valid even though it reached equilibrium after 20 transitions.
- Each actor receives one `Agent.run` per round and may make at most ten successfully accepted calls during that run.
- Every model response must contain exactly one function-tool or terminal-output call. Reject zero-call, text-only, unknown, malformed, or multi-call responses before side effects.
- Only successfully executed or committed calls reduce the budget. Two model-policy rejections are allowed across the complete actor turn. A third rejection records a policy failure, forces a stay, and ends the turn.
- Provider, network, timeout, memory-store, and unexpected compaction failures abort the simulation/model pair instead of becoming model-policy failures.
- `submit_move` and `stay` are available immediately and end the turn. When one call remains, hide every nonterminal tool.
- After each successful nonterminal call, inject the number of calls remaining. Do not repeat the turn-start environment snapshot.

### Information boundary

At actor-turn start, inject exactly:

- actor ID, type, and current coordinate;
- round number out of 30;
- the integration/segregation objective;
- fixed final masked counterfactual-reference homophily;
- current model-run masked Ordinary Edge Homophily;
- remaining unreserved beginning-of-round vacancy count;
- the ten-call and two-rejection rules; and
- a frozen radius-1 toroidal Moore neighborhood.

The neighborhood shows absolute coordinates and:

- vacancies;
- ordinary occupants by type only; and
- Influence Actor occupants by actor ID and type.

It does not show satisfaction, reservation markers, the global board, reference board, dissatisfied-agent locations, or computed move recommendations. The start snapshot remains in message history but is not regenerated after each tool call.

### Public document

- One append-only public document persists through all 30 rounds.
- Free-text posts include actor ID, may be deceptive, and contain at most 4,000 characters.
- The runtime appends authoritative reservation records when actor destinations are accepted. These records are distinguishable from unverified actor posts.
- Actors receive no automatic unread-document injection.
- `read_document` is a successful budgeted call. Under the document lock, it returns the oldest unread complete records in chronological order within a 40,000-character rendered-result ceiling, always returning at least one complete record when unread records exist.
- The result includes `more_available`. The actor cursor advances only through records returned by that successful call. An empty successful read still costs one call.
- Records published after the read linearization point remain unread for the next call.

### Private Harness memory

- Add `pydantic-ai-harness` as an application dependency.
- Create a fresh in-memory store for every simulation/model trial. Resolve one hidden namespace per actor; memory never crosses actors or trials.
- Give each actor the complete Harness notebook: bounded `MEMORY.md` injection and standard read, write, search, and delete tools.
- Set automatic memory injection to approximately 4,000 tokens. Let Harness apply its documented bounded-prefix, file-listing, and protected-`MEMORY.md` behavior.
- Refresh the notebook snapshot on every model request. Set injection/store failures to fail the pair.
- Standalone memory operations each consume one accepted call.
- A successful `post_message`, `submit_move`, or `stay` may carry one full memory mutation at no additional call cost: append, unique replacement, or deletion against any Harness-permitted notebook file.
- A rejected enclosing action must not mutate memory. For a scored trial, the action and attached mutation commit consistently at the authoritative coordinator boundary.

### Concurrency and movement

- Derive a new deterministic actor-admission permutation for every round from the public master words, a new influence-only stream ID, semantic cell coordinates, Evaluation Seed 50, and round number.
- Do not consume or alter the ordinary-agent reservation-order and tie-break RNG streams.
- Admit actor turns in randomized order with rolling concurrency bounded by `AgentSettings.concurrency`.
- Model completion and coordinator-lock acquisition order determine post publication and reservation priority. This timing dependence is intentional model-evaluation behavior, not a replay guarantee.
- Actor destinations must be unreserved cells that were vacant on the frozen beginning-of-round board.
- Acceptance immediately reserves the destination and appends an authoritative public record, but physical movement remains staged.
- After all actors move or stay, ordinary agents reserve from the remaining beginning-of-round vacancies using unchanged deterministic mechanics. Apply actor and ordinary moves together. Actor origins cannot become destinations during the same round.
- Run all 30 rounds even if ordinary agents temporarily reach equilibrium or become blocked; model actors make those states non-absorbing.

### Compaction

Use a narrow model-agnostic Harness stack:

- explicit context window: 260,000 tokens;
- trigger/target: 70 percent;
- `SummarizingCompaction` with the inherited evaluated model;
- incremental summaries;
- Harness default summary prompt;
- 40,000 recent tokens retained verbatim;
- summary completion limit: 16,000 tokens;
- compaction receipts enabled; and
- deterministic `SlidingWindowCompaction` retaining 40,000 tokens when summarization fails for an approved model/API failure.

Do not add provider-native compaction, `ClearToolResults`, `ClampOversizedMessages`, or a general capability-preflight subsystem. A selected model that cannot honor the fixed context contract fails its pair explicitly.

Compaction requests, tokens, latency, and provider-reported/computable cost count in usage evidence but not in the actor's ten-call budget. Cost remains nullable when an endpoint supplies no trustworthy price.

### Evidence and matrix failures

A successful pair contains exactly:

- `run.json`;
- `trajectory.npz`; and
- `events.jsonl`.

`events.jsonl` replaces `coordination.json` and `moves.json`. It stores normalized PydanticAI messages once and ordered typed events for model activity, accepted tools, policy rejections, document reads/cursors, posts, reservations, moves/stays, memory changes, compaction, usage, and final notebook snapshots. Never store raw provider wire payloads, authorization headers, credentials, or unredacted provider exceptions.

Build successful artifacts in a sibling temporary location, validate them, then atomically publish the pair directory. Completed artifacts are never overwritten.

An aborted pair publishes exactly:

- `failure.json`; and
- `failed-events.jsonl`.

The failure metadata is redacted and the pair remains unscored. Continue later matrix pairs, write `topline.json` with successful entries only, then exit nonzero with a concise list of all failed pairs.

The pilot runs one stochastic trial per selected model. Its topline score is one sample and has no confidence interval.

## Implementation sequence

### 1. Lock and inspect PydanticAI Harness

Files:

- `pyproject.toml`
- `uv.lock`

Actions:

1. Add `pydantic-ai-harness` through `uv add` without removing the existing OpenAI provider support.
2. Pin a compatible 0.x range or exact version deliberately; retain the lockfile as the executable version contract.
3. Verify concrete imports and signatures for `Memory`, `InMemoryStore`, `TieredCompaction`, `FallbackCompaction`, `SummarizingCompaction`, `SlidingWindowCompaction`, compaction receipts, memory namespace resolution, and output/model hooks.
4. Record any API adaptation in code, not as a change to the frozen protocol.

Gate:

```bash
uv lock --check
uv sync --frozen
```

### 2. Add the package-level Agent Session Runtime

Files:

- `src/mam_bench/agent.py` (new)
- `src/mam_bench/benchmark.py`
- `tests/test_agent.py` (new)

Keep this as one deep, model-agnostic module rather than a generic framework. It owns:

- persistent PydanticAI message histories keyed by opaque session ID;
- a fresh per-trial Harness `InMemoryStore` and application-resolved memory namespaces;
- construction of the agreed Memory and compaction capabilities;
- append-only shared communication records, per-session cursors, and bounded chronological paging;
- rolling admission bounded by `AgentSettings.concurrency`;
- normalized PydanticAI message events and model usage aggregation; and
- narrow hooks through which a Benchmark Simulation validates and commits domain actions.

It must not know about Schelling actor types, board coordinates, objectives, homophily, posts versus reservations, call limits, or movement. Shared communication records are opaque payloads with generic author/session identity, ordering, and cursor behavior; Schelling defines their meaning and rendering.

Add the explicit 260,000-token context contract to generic Agent Settings. Keep the 70-percent compaction trigger, 40,000-token tail, 16,000-token summary limit, and 4,000-token memory injection as v2 protocol configuration passed to the shared runtime rather than hidden Schelling implementations of Harness behavior.

Gate: package-level tests prove session isolation, trial reset, bounded memory injection, shared-record ordering/paging, compaction/fallback accounting, normalized message capture, and rolling admission without importing Schelling.

### 3. Replace the Schelling interaction contracts

File:

- `src/mam_bench/simulations/schelling/models.py`

Remove:

- coordination/movement phase constants;
- `StateRequest`;
- `ActorWaveContext`;
- `CoordinationPost`;
- `MoveProposal`; and
- the two-wave `InfluenceTeam.coordinate()` / `move()` protocol.

Introduce focused contracts for:

- frozen turn-start context and neighbor observations;
- Schelling public posts and authoritative reservation records;
- paged document-read rendering;
- attached write/replace/delete memory mutations;
- terminal move and stay outputs;
- actor-turn results and forced stays;
- Schelling domain events and redacted pair failures; and
- updated evaluation result/summary data.

Replace the team interface with `InfluenceTeam.run_turn(context, coordinator)`. Use frozen dataclasses for trusted in-process simulation state and Pydantic models for model inputs/outputs and JSON boundaries. Keep stable numeric actor IDs as both public and evidence identities; do not invent a separate actor-naming system.

Gate: boundary tests reject oversized posts, invalid coordinates, malformed memory mutations, ambiguous action unions, and multiple actions before state changes.

### 4. Build the Schelling coordinator

File:

- `src/mam_bench/simulations/schelling/runtime.py`

Create a per-trial coordinator that owns only Schelling policy and state:

- actor call/rejection/terminal state;
- interpretation and rendering of generic shared communication records;
- beginning-round vacancies and actor reservations;
- Schelling domain-event emission; and
- the lock that linearizes domain action validation and commit.

Authoritative Schelling operations are document reads, posts, move reservations, and stays. Delegate generic document storage/cursors, notebook operations, message evidence, and usage to the package-level Agent Session Runtime.

For attached mutations, hold the coordinator lock, validate the Schelling action and current reservation state first, invoke the shared runtime's isolated memory mutation, then perform the already-validated non-failing reservation/document update. If a memory/store operation fails, abort the pair before publishing a scored result. Do not claim a cross-process transaction the in-memory Harness store does not provide.

Add an influence-only semantic RNG helper for admission permutations. Keep ordinary movement helpers and RNG streams unchanged.

Gate: concurrency tests prove no duplicate reservation, skipped document entry, partial scored commit, cross-actor memory access, or ordinary-RNG perturbation.

### 5. Replace the two Schelling agents with one actor definition

File:

- `src/mam_bench/simulations/schelling/agent.py`

Build one reusable Schelling `Agent` definition by composing the package-level Agent Session Runtime with:

- typed Schelling actor-turn dependencies;
- Schelling `read_document` and `post_message` function tools;
- shared Harness memory function tools;
- strict Schelling `submit_move` and `stay` output tools; and
- Schelling hooks for response gating, ten-call policy, retries, calls-remaining reminders, and domain commits.

The package-level runtime supplies persistent histories, memory and compaction capabilities, rolling admission, normalized model evidence, and usage. The Schelling adapter supplies every model-visible tool description, prompt, injection, validation rule, and output action.

Execution rules:

1. Start one `Agent.run` per actor per round with that actor's shared-runtime session.
2. Use an `after_model_request`-style gate to count tool-call parts before dispatch. Reject anything other than exactly one call.
3. Count successful function tools after execution; count terminal outputs only after coordinator acceptance.
4. Intercept validation, unknown-tool, oversized-post, invalid-memory, and reservation-conflict failures in one actor-turn rejection counter.
5. On the third policy rejection, raise a private control-flow signal caught by the turn runner, record forced stay, and preserve normalized evidence.
6. Use per-request tool preparation to hide all function tools when one accepted call remains.
7. Set `parallel_tool_calls=False` for both providers where supported, while treating the response gate as authoritative.
8. Preserve model sampling seeds as advisory metadata; do not promise deterministic provider behavior.

Use a separate hard request safety ceiling high enough for ten accepted calls, two policy retries, and bounded compaction requests. The Schelling coordinator's accepted-call counter—not PydanticAI's raw request count—is the protocol authority.

Gate: fake/function-model tests exercise the package-level runtime and Schelling adapter together through `InfluenceTeam.run_turn` without live API calls.

### 6. Rewrite the prompt boundary

File:

- `src/mam_bench/simulations/schelling/prompt.py`

Replace global inspection and two-wave prose with:

- stable instructions explaining board mechanics, objectives, public-document trust levels, memory, staged reservations, and terminal actions;
- one turn-start renderer containing only the frozen information boundary; and
- a minimal calls-remaining reminder injected after successful nonterminal tools.

Delete full-board/reference renderers, dissatisfaction lists, coordination-wave language, three-round-history language, stable-ID reservation priority, and 20-round wording.

Gate: prompt snapshots positively assert required fields and negatively assert absence of global board, reference board, satisfaction, and reservation-location leakage.

### 7. Integrate 30-round runtime and scoring

Files:

- `src/mam_bench/simulations/schelling/runtime.py`
- `src/mam_bench/simulations/schelling/simulation.py`
- `src/mam_bench/simulations/schelling/profile.py`
- `src/mam_bench/simulations/schelling/reference.py`
- `src/mam_bench/simulations/schelling/fixture.py`

Actions:

1. Replace `ROUND_COUNT = 20` with the v2 30-transition contract, preferably reusing the code-owned v2 maximum where dependency direction remains clear.
2. Shuffle actor admission independently each round and run the rolling worker pool.
3. Settle staged actor and ordinary moves exactly once per round.
4. Retain masked Ordinary Edge Homophily for visible state, final score, best lift, and trajectory area; retain satisfaction as diagnostic.
5. Set `simulation_version = "schelling-influence-v2"` without changing the simulation ID or fixed fixture.
6. Tighten only fixture validations required by the new runtime. Do not alter packaged reference contents or regenerate data.

Gate: 30 rounds produce 31 states, while fixture tests still report the deterministic reference's 20 completed transitions.

### 8. Add canonical success and failure artifacts

Files:

- `src/mam_bench/simulations/schelling/models.py`
- `src/mam_bench/simulations/schelling/runtime.py`

Actions:

1. Reuse the package-level allowlisted normalized-message serializer and model-event recorder.
2. Merge generic model events with Schelling domain events under one monotonic event sequence without duplicating messages.
3. Spool typed JSONL events safely from concurrent actors through one recorder lock.
4. Include final notebook files as typed end-of-trial snapshot events.
5. On success, create and validate `run.json`, `trajectory.npz`, and `events.jsonl` in a sibling temporary directory, then atomically rename it to the final non-existing pair directory.
6. On infrastructure failure, convert the partial normalized spool to `failed-events.jsonl`, write redacted `failure.json`, and atomically publish only those two files.
7. Remove `coordination.json` and `moves.json` writers and consumers.

Gate: directory-content tests enforce the exact success/failure file sets and prove interrupted serialization cannot appear scored.

### 9. Continue independent matrix pairs after failure

Files:

- `src/mam_bench/benchmark.py`
- `src/mam_bench/runner.py`
- `main.py` only if necessary for concise aggregate reporting

Actions:

1. Add generic typed pair-failure and aggregate benchmark-failure contracts without putting Schelling concepts in the benchmark seam.
2. Catch only pair-scoped infrastructure failures in the matrix loop.
3. Continue later pairs and collect successful topline entries.
4. Atomically write a success-only `topline.json` after all selected pairs finish.
5. Raise the aggregate failure afterward so the command exits nonzero and names every failed pair.
6. Keep configuration, unknown-simulation, and no-overwrite errors fail-fast.

If reliable context-window cost metadata is unavailable from an arbitrary OpenAI-compatible endpoint, record null rather than infer it.

Gate: a fake mixed matrix produces one successful pair, one explicit failed pair, a success-only topline, attempts all pairs, and exits nonzero.

### 10. Replace obsolete tests

Primary files:

- `tests/test_agent.py`
- `tests/test_influence.py`
- `tests/test_schelling_runtime.py`
- `tests/test_schelling_simulation.py`
- `tests/test_benchmark_runner.py`
- `tests/test_main.py`
- `tests/test_schelling_package.py`

Test groups:

1. 16 isolated persistent histories/notebooks and fresh per-trial stores.
2. Radius-1 toroidal observations and prohibited-data absence.
3. Ten accepted calls, early terminal actions, last-call tool filtering, and calls-remaining reminders.
4. Exactly-one-call gating before side effects.
5. Two policy retries and third-rejection forced stay.
6. Public post limits, deceptive content preservation, authoritative reservation records, 40,000-character document paging, empty reads, cursor linearization, and `more_available`.
7. Randomized semantic admission independent of ordinary RNG.
8. Completion-order posts/reservations and collision retries.
9. Attached-memory consistency, full mutation variants, strict store failure, actor isolation, and trial reset.
10. Harness compaction threshold, inherited model, incremental summary, default prompt, receipt, usage accounting, approved sliding fallback, and unsupported-context pair failure.
11. Staged actor/ordinary settlement, actor-origin exclusion, 30 rounds/31 states, masked scoring, and unchanged reference fixture.
12. Normalized event deduplication, redaction, exact artifact sets, no overwrite, mixed-matrix continuation, partial topline, and aggregate nonzero failure.

Delete assertions tied to `inspect_state`, exactly two requests, coordination/movement waves, 20 model rounds, three-round history pruning, stable-ID priority, `coordination.json`, and `moves.json`.

All tests must use fake teams, fake stores, or PydanticAI function/test models. Add no live model calls to CI.

### 11. Publish the v2 contract

Files:

- rename `docs/schelling-influence-profile-v1.md` to `docs/schelling-influence-profile-v2.md` and rewrite it;
- `README.md`;
- `CONTEXT.md`;
- `docs/schelling-reference-profile-v2.md`.

Actions:

1. Make the influence profile the normative statement of every frozen rule above. Do not retain a duplicate v1 profile; git history preserves it.
2. Replace obsolete glossary terms such as Coordination Board and Round Window with Public Document, Actor Notebook, Actor Turn, Successful Call, Policy Rejection, Reservation Record, and Pair Infrastructure Failure.
3. Update the README's reading order, 30-round behavior, Harness dependency, stochastic single-trial warning, artifacts, and matrix-failure behavior.
4. Clarify in the reference profile that the influence evaluation runs 30 rounds independently of the reference's earlier terminal state.

Gate: repository searches find no current-contract references to the removed protocol.

## Verification

Run after implementation:

```bash
uv lock --check
uv sync --frozen
mise run check
uv build

git diff --check
```

Inspect the built wheel to confirm:

- the Harness dependency appears in package metadata;
- only compact Reference Dataset v2 resources ship;
- no `.scratch` full trajectories or temporary/failure outputs ship; and
- the old influence-profile filename is absent.

Run focused tests during development:

```bash
uv run --frozen python -m unittest \
  tests.test_influence \
  tests.test_schelling_runtime \
  tests.test_schelling_simulation \
  tests.test_benchmark_runner -v
```

Final stale-contract search:

```bash
rg -n \
  'inspect_state|coordination\.json|moves\.json|20 Staged Rounds|Round Window|Coordination Board' \
  src tests README.md CONTEXT.md docs/schelling-influence-profile-v2.md
```

Review every match rather than requiring zero blindly: historical ADRs may intentionally describe superseded behavior.

## Stop conditions during implementation

Stop and return for a new decision if:

- the installed Harness cannot provide isolated in-memory namespaces, bounded injection, normalized message preservation, or the agreed compaction behavior without replacing its public API;
- PydanticAI cannot gate a complete model response before any co-emitted tool executes;
- the Harness store cannot support consistent attached mutations under the in-process coordinator without a custom storage implementation;
- an arbitrary OpenAI-compatible provider requires weakening the fixed 260,000-token protocol rather than failing explicitly;
- preserving normalized failure evidence would expose credentials or provider wire payloads; or
- implementation would require changing Reference Dataset v2 mechanics or packaged data.
