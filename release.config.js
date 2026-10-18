/** Release Rust packages to crates.io and publish matching GitHub release notes. */
export default {
  branches: ["main"],
  repositoryUrl: "https://github.com/cmeister2/magicextract",
  tagFormat: "${version}",
  plugins: [
    "@semantic-release/commit-analyzer",
    [
      "@semantic-release/release-notes-generator",
      { preset: "conventionalcommits" },
    ],
    [
      "@semantic-release/exec",
      {
        prepareCmd: "node scripts/prepare-release.mjs ${nextRelease.version}",
        publishCmd: "cargo publish --locked --allow-dirty",
      },
    ],
    "@semantic-release/github",
  ],
};
