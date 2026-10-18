import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import test from "node:test";
import releaseConfig from "./release.config.js";

// Follow semantic-release's dependency graph rather than a separately installed writer.
const require = createRequire(import.meta.url);
const requireFromSemanticRelease = createRequire(require.resolve("semantic-release"));
const { generateNotes } = await import(
  pathToFileURL(
    requireFromSemanticRelease.resolve("@semantic-release/release-notes-generator"),
  ).href
);
const { analyzeCommits } = await import(
  pathToFileURL(
    requireFromSemanticRelease.resolve("@semantic-release/commit-analyzer"),
  ).href
);

const repositoryUrl = "https://github.com/cmeister2/magicextract";
const logger = { log() {}, error() {} };

test("renders complete release notes through semantic-release's actual generator", async () => {
  const plugin = releaseConfig.plugins.find(
    (entry) => Array.isArray(entry) && entry[0] === "@semantic-release/release-notes-generator",
  );
  assert.ok(plugin, "release configuration must include the notes generator");
  const commits = [
    { hash: "a".repeat(40), message: "feat: find perspective card outlines" },
    { hash: "b".repeat(40), message: "fix: avoid duplicate card crops\n\nCloses #42" },
    {
      hash: "c".repeat(40),
      message:
        "feat: return upright card crops\n\nBREAKING CHANGE: Crop dimensions now follow detected card orientation.",
    },
  ];
  const notes = await generateNotes(plugin[1], {
    cwd: process.cwd(),
    env: {},
    logger,
    options: { repositoryUrl },
    branch: { name: "main" },
    lastRelease: { version: "1.2.3", gitTag: "1.2.3" },
    nextRelease: { version: "2.0.0", gitTag: "2.0.0" },
    commits,
  });

  assert.match(notes, /### Features/);
  assert.match(notes, /### Bug Fixes/);
  assert.match(notes, /### .*BREAKING CHANGES/);
  assert.match(notes, /Crop dimensions now follow detected card orientation\./);
  assert.match(notes, /find perspective card outlines/);
  assert.match(notes, /avoid duplicate card crops/);
  assert.match(notes, /return upright card crops/);
  assert.ok(notes.includes(`${repositoryUrl}/compare/1.2.3...2.0.0`));
  assert.ok(notes.includes(`${repositoryUrl}/issues/42`));
  for (const commit of commits) {
    assert.ok(notes.includes(`${repositoryUrl}/commit/${commit.hash}`));
  }
});

test("keeps the analyzer's default release rules", async () => {
  assert.ok(releaseConfig.plugins.includes("@semantic-release/commit-analyzer"));
  for (const [message, expected] of [
    ["fix: retain the full card border", "patch"],
    ["perf: reduce edge detection allocations", "patch"],
    ["feat: extract card crops", "minor"],
    ["feat: revise crop output\n\nBREAKING CHANGE: Crop output dimensions changed.", "major"],
    ["docs: clarify extraction scope", null],
    ["chore: refresh release tooling", null],
  ]) {
    const actual = await analyzeCommits({}, {
      cwd: process.cwd(),
      logger,
      commits: [{ hash: "d".repeat(40), message }],
    });
    assert.equal(actual, expected, message);
  }
});
