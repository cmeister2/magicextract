# Releasing

Release tooling is a private Node project alongside the Rust crate. Use
**Node 24.15.0 or newer**; CI selects Node 24.

The crate package contains Rust sources, the extraction example, standalone
procedural tests, and documentation. Saved image corpora, their evaluator, and
authoring tools stay in the repository; CI tests them before publication.

## Validate the tooling

```sh
npm ci
npm test
```

Run both after changing release configuration or dependencies. Update
`package-lock.json` alongside `package.json`.

Keep `conventional-changelog-conventionalcommits` pinned to `10.4.0` and the
`@semantic-release/release-notes-generator` override at `15.0.0-beta.3` until a
replacement passes the regression. Preset 10 uses JavaScript render functions,
which require writer 9 or newer; semantic-release 25's ordinary dependency graph
can select generator 14 with writer 8. The override supplies a compatible writer.

`release.config.test.mjs` reads the configured generator options and resolves the
generator through semantic-release's actual dependency graph. It checks feature,
fix, and breaking-change sections plus compare, issue, and commit links. Separate
cases check the analyzer's version rules and synchronized version preparation.

The pinned tooling has a transitive [`braces` denial-of-service advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm).
Review upstream fixes when updating release dependencies; this dependency belongs
to the private release tooling, outside the published Rust dependency graph.

## Configure publication

Set the repository's `CARGO_REGISTRY_TOKEN` Actions secret to a crates.io token
authorized to publish this crate. Actions supplies `GITHUB_TOKEN`; the release
job grants the permissions needed for GitHub releases and related issue comments.

Only **pushes to `main` or `master`** run semantic-release, after Rust checks,
MSRV verification, coverage, and release-tooling regressions pass. The release
checkout fetches complete history and tags. The workflow runs `npm ci` and
`npm test` again before invoking semantic-release.

`release.config.js` publishes to crates.io and creates a GitHub release. Tags are
plain SemVer, such as `1.2.3`. Preserve existing release tags and history so
semantic-release can identify the previous release.

## Commit messages

| Message | Release |
| --- | --- |
| `fix:` or `perf:` | Patch |
| `feat:` | Minor |
| `BREAKING CHANGE:` footer | Major |
| Ordinary `docs:`, `test:`, `ci:`, or `chore:` | None |

The analyzer uses its default preset. Use a `BREAKING CHANGE:` footer with an
explanation for breaking changes; preserve meaningful messages when merging.

## Version preparation

Keep `Cargo.toml` and the crate's `Cargo.lock` entry at **`0.0.0`** in source.
During preparation, semantic-release calls `scripts/prepare-release.mjs` with the
selected version. The script validates both sentinels before replacing them and
preserves dependency versions. Release checkout changes stay out of source commits.

When a push produces no release, CI injects `0.0.<github.run_id>` into both files
for the subsequent `cargo build --release --locked` in that checkout.
