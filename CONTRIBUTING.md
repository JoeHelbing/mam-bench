# Contributing

MAM-Bench is an early, solo-maintained research project. Its interfaces, suites, and scores can change. Issues and pull requests are reviewed as time permits; opening one does not guarantee a response or acceptance.

For a bug, include a small reproduction, expected and actual behavior, and the version or commit tested. For a proposed change, explain the problem and its effect on benchmark behavior. Do not post credentials, private model conversations, or paid-provider logs.

Target `dev` for changes. Use a Conventional Commit PR title such as `fix: correct case validation` or `feat(schelling): add a scenario`; feature PRs are squash-merged. Keep the change focused and run the local checks from [README.md](README.md#verify-without-model-calls). Say which checks you ran and note any that you could not run. Those checks do not require model calls; running a benchmark can incur provider charges.

Maintainers promote `dev` to `main` with a normal merge, not a squash merge. After promotion, review the release-please PR against the promoted commits before merging it, then merge `main` back into `dev`. A release PR merge creates the GitHub Release; promotion alone does not.
