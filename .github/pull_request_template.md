<!-- Target dev for ordinary changes. Main accepts dev promotions and release PRs. -->

## Why

<!-- Describe the problem, expected behavior, and why the change is needed.
Link the issue or reproduction when available. -->

## What changed

<!-- Summarize the changes a reviewer should focus on. Call out effects on
simulation rules, scoring, configuration, provider requests, or saved artifacts. -->

## Test coverage

<!-- Follow CONTRIBUTING.md#testing-changes. For documentation-only changes,
explain why behavioral test changes are unnecessary. -->

- Behavior protected, and the scenario or existing tests that cover it:
- Source of the expected result, and regression-failure evidence where applicable:
- Justification for removed tests, changed expectations, assertions, or skips:

## End-to-end validation with actual artifacts

<!-- Follow CONTRIBUTING.md#end-to-end-validation-with-actual-artifacts.
Describe the prototype or trial run, not just the automated test results.
Identify real captured inputs, fresh live inputs, and synthetic fixtures accurately.
If this validation is not applicable or could not run, explain why and what remains
unverified. Share only artifacts that are appropriate for a public PR. -->

- Question investigated and criteria for a useful result:
- Input source, sample size, selection, and coverage of real variations:
- Procedure, commands, configuration, and revision tested:
- Artifacts produced and inspected, with shareable references or reproduction steps:
- Findings: successful and failed cases, measurements, and concrete examples:
- Refinements made from those findings and results after rerunning:
- Remaining limitations or validation not performed:

## Verification

<!-- Commands and requirements are in CONTRIBUTING.md#verification-checks.
Check only items actually run and passed. Report results and explain unchecked
items, including checks intentionally skipped for documentation-only changes.
Do not include credentials, private model conversations, or paid-provider logs. -->

- [ ] Full offline unittest suite
- [ ] Ruff lint
- [ ] Ruff format check
- [ ] Strict Pyright
- [ ] git diff --check

Results, checks not run, and remaining limitations:
