# Contributing

MAM-Bench is an early, solo-maintained research project. Its interfaces, suites, and scores can change. Issues and pull requests are reviewed as time permits; opening one does not guarantee a response or acceptance.

For a bug, include a small reproduction, expected and actual behavior, and the version or commit tested. For a proposed change, explain the problem and its effect on benchmark behavior. Do not post credentials, private model conversations, or paid-provider logs.

Target `dev` for changes. Use a Conventional Commit PR title such as `fix: correct case validation` or `feat(schelling): add a scenario`; feature PRs are squash-merged. Keep the change focused and complete the [verification checks](#verification-checks). Say which checks you ran and note any that you could not run. Those checks do not require model calls; running a benchmark can incur provider charges.

## Verification checks

Run commands from the repository root. Use Python 3.14.7 and uv 0.12.7, as pinned
in [mise.toml](mise.toml). If the environment is not ready, prepare the locked
runtime dependencies first:

```fish
uv python install 3.14.7
uv sync --frozen --no-dev
```

For code or test changes, run all of the following before opening or updating a
PR. These commands match the test, lint, format, and type checks in
[CI](.github/workflows/ci.yml), plus a local whitespace check:

```fish
# Run the full offline suite.
uv run --no-sync python -m unittest discover -s tests -v

# Check lint and formatting with the pinned Ruff version.
uvx --from ruff==0.16.5 ruff check src tests main.py
uvx --from ruff==0.16.5 ruff format --check src tests main.py

# Run strict type checking against the project environment; requires Node.js/npm.
npx --yes pyright@1.1.413 --pythonpath .venv/bin/python

# Check the patch for whitespace errors.
git diff --check
```

Use installed tools at the same versions if available; `uvx` and `npx` may download
their pinned tools. The tests use synthetic provider responses and require no
model credentials or paid inference calls. Real-provider benchmark evaluations
are separate and are not required for an ordinary PR.

During development, run the affected scenarios on their own before the full
suite. For documentation-only changes, inspect the rendered structure, local
links, and any documented commands, and run `git diff --check`; explain why the
application checks were not rerun. In every PR, report checks actually run and
their results, with reasons for skipped or unavailable checks. An unchecked box
is not a successful check. GitHub also verifies the Conventional Commit PR title
and runs the application checks on the published revision.

## Testing changes

This is the testing policy for both human and agent contributions. Keep a small,
readable suite that protects benchmark behavior and allows implementation changes.
The goal is trustworthy failure detection, rather than a test for every function.

### Choose a behavior and a boundary

Before editing tests, name the behavior at risk, its expected outcome, and the
interface through which a caller observes it. Start with the existing scenarios
for that behavior. Extend one when it can express the change clearly; add a scenario
when it protects a distinct failure. If existing checks already cover the change,
explain that coverage instead of adding a duplicate.

Prefer these boundaries, keeping real application code in the path:

| Behavior | Preferred scenario |
| --- | --- |
| Configuration and serving limits | Load configuration and resolve setup against an HTTP discovery response. |
| Agent actions and simulation rules | Step or evaluate a simulation through real model tools; inspect accepted actions and saved states. |
| Provider integration | Send requests through the real SDK using a mocked HTTP transport; inspect serialized JSON and the resulting action. |
| Run completion and failure | Execute a mixed suite or the CLI; inspect results, retained artifacts, cancellation, and whether a final total is published. |

Retain a focused lower-level check when it catches a distinct failure that these
scenarios would miss, such as a numerical boundary or an action-reservation race.
Assertions about helper names, incidental call order, SDK keyword dictionaries,
or equivalent JSON Schema ordering need a specific compatibility reason.

### Keep expected results independent

Expected values must come from a requirement, worked example, invariant, or trusted
reference. Copying production logic into the expected-value calculation, or recording
the current output alone, does not establish that behavior is correct. When
preserving observed behavior during a refactor, distinguish that characterization
from an independently established requirement.

For score replay, derive scored populations from the configured case and initial
population or observed controlled participants. Derive participation and the first
revolution cycle from raw saved states. Check the result's identity lists,
termination, and score against those expectations; using its own identity lists
and termination labels to calculate the expected score can hide a shared defect.
Keep hand-counted examples for exclusions, jailed participation, signed scores,
and threshold boundaries alongside broader replay scenarios.

For a bug fix, run the regression check before the fix and confirm that it fails
at the intended assertion, rather than during import or setup. Then confirm that
the same check passes after the fix. For other important assertions, a targeted
deliberate fault can demonstrate detection power. A separate agent's agreement or
a higher coverage number does not replace an independently justified expectation.

### Consolidate without hiding failures

One scenario should describe one coherent behavior. Use named `subTest` variants
for related providers, roles, settings, or boundary values so failures identify the
case. Keep unrelated behaviors in separate scenarios; combining them solely to
reduce the method count makes diagnosis harder. Share small setup helpers when they
remove repeated wiring, while keeping the scenario and expected outcomes visible.

When consolidating or retiring a test, identify where each distinct behavior is
still checked. Preserve meaningful protection for paired worlds, invalid-action
recovery, cancellation, incomplete runs without final scores, and partial artifact
durability. Review removed assertions, changed expectations, snapshot updates, and
skips explicitly. Test count and coverage increases are not acceptance criteria;
readability, useful fault detection, runtime, and reliability matter together.

### Run checks in a controlled environment

Keep the normal suite offline and deterministic. Mock external providers at the
HTTP boundary, using synthetic credentials and responses. Exercise real SDK
serialization and application parsing. For failures that cannot be induced at
that boundary, keep injection narrow and assert the resulting public behavior.

Control randomness and time where relevant, use temporary output directories,
and restore environment variables, patches, clients, and other shared state.
Prefer explicit synchronization to timing sleeps in concurrency scenarios.
Run changed scenarios alone and then with the relevant suite to catch state
pollution. Repeated passing runs can expose flakiness but do not prove its absence.
Before submitting, run the [verification checks](#verification-checks)
and report any checks that could not run. Real provider evaluations remain separate
because they require credentials and can incur charges.

Keep exploratory checks and deliberate-fault experiments in Git-ignored `.scratch/`;
put accepted regression scenarios in the normal test suite. In the PR, summarize
the protected behavior, the source of the expected outcome, failure-detection
evidence where applicable, and the verification performed. Explain why removed or
rewritten checks retain the required protection.

For background, read Google's [unit-testing guidance](https://abseil.io/resources/swe-book/html/ch12.html),
Meta's [TestGen-LLM study](https://arxiv.org/abs/2402.09171), and its
[mutation-guided test-generation study](https://arxiv.org/abs/2501.12862).

## Promotion and release

The promotion workflow opens a `dev` to `main` PR when `dev` has unpromoted commits. GitHub updates the open PR as more changes land on `dev`; after it is merged, the workflow opens a new one on the next `dev` push with unpromoted commits. Feature PRs into `dev` are squash-merged so their Conventional Commit titles become the change commits. Merge the promotion PR with a **merge commit**, not squash, so release-please sees each `feat:` or `fix:` commit from `dev`. The `main` ruleset requires merge commits; the `dev` ruleset allows both methods because syncing `main` back into `dev` also needs a merge commit.

After promotion, release-please opens a **separate** version/changelog PR against `main`. Review it against the promoted commits before merging it. Once the release PR is merged, merge `main` back into `dev` with a merge commit, not squash. The release PR merge creates the GitHub Release; promotion alone does not.
