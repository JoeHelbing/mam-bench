# Contributing

MAM-Bench is an early, solo-maintained research project. Its interfaces, suites, and scores can change. Issues and pull requests are reviewed as time permits; opening one does not guarantee a response or acceptance.

For a bug, include a small reproduction, expected and actual behavior, and the version or commit tested. For a proposed change, explain the problem and its effect on benchmark behavior. Do not post credentials, private model conversations, or paid-provider logs.

Target `dev` for changes. Use a Conventional Commit PR title such as `fix: correct case validation` or `feat(schelling): add a scenario`; feature PRs are squash-merged. Keep the change focused and run the local checks from [README.md](README.md#verify-without-model-calls). Say which checks you ran and note any that you could not run. Those checks do not require model calls; running a benchmark can incur provider charges.

The promotion workflow opens a `dev` to `main` PR when `dev` has unpromoted commits. GitHub updates the open PR as more changes land on `dev`; after it is merged, the workflow opens a new one on the next `dev` push with unpromoted commits. Feature PRs into `dev` are squash-merged so their Conventional Commit titles become the change commits. Merge the promotion PR with a **merge commit**, not squash, so release-please sees each `feat:` or `fix:` commit from `dev`. The `main` ruleset requires merge commits; the `dev` ruleset allows both methods because syncing `main` back into `dev` also needs a merge commit.

After promotion, release-please opens a **separate** version/changelog PR against `main`. Review it against the promoted commits before merging it. Once the release PR is merged, merge `main` back into `dev` with a merge commit, not squash. The release PR merge creates the GitHub Release; promotion alone does not.
