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
