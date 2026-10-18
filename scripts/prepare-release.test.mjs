import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { prepareRelease } from "./prepare-release.mjs";

const manifest = '[package]\nname = "magicextract"\nversion = "0.0.0"\n';
const lockfile =
  'version = 4\n\n[[package]]\nname = "another-crate"\nversion = "0.0.0"\n\n[[package]]\nname = "magicextract"\nversion = "0.0.0"\n';

async function fixture(context, lockContents = lockfile) {
  const directory = await mkdtemp(join(tmpdir(), "magicextract-release-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(join(directory, "Cargo.toml"), manifest);
  await writeFile(join(directory, "Cargo.lock"), lockContents);
  return directory;
}

test("updates both crate versions while preserving dependency versions", async (context) => {
  const directory = await fixture(context);
  await prepareRelease("1.2.3", directory);
  assert.equal(
    await readFile(join(directory, "Cargo.toml"), "utf8"),
    manifest.replace('version = "0.0.0"', 'version = "1.2.3"'),
  );
  assert.equal(
    await readFile(join(directory, "Cargo.lock"), "utf8"),
    lockfile.replace('name = "magicextract"\nversion = "0.0.0"', 'name = "magicextract"\nversion = "1.2.3"'),
  );
});

test("accepts CI build versions", async (context) => {
  const directory = await fixture(context);
  await prepareRelease("0.0.12345678", directory);
  assert.match(await readFile(join(directory, "Cargo.toml"), "utf8"), /version = "0\.0\.12345678"/);
});

test("leaves both files untouched if a sentinel is absent", async (context) => {
  const releasedLockfile = lockfile.replace('name = "magicextract"\nversion = "0.0.0"', 'name = "magicextract"\nversion = "1.2.3"');
  const directory = await fixture(context, releasedLockfile);
  await assert.rejects(prepareRelease("2.0.0", directory), /sentinel/);
  assert.equal(await readFile(join(directory, "Cargo.toml"), "utf8"), manifest);
  assert.equal(await readFile(join(directory, "Cargo.lock"), "utf8"), releasedLockfile);
});

test("rejects invalid versions before touching files", async (context) => {
  const directory = await fixture(context);
  for (const version of [undefined, "", "0.0.0", "1.2", "v1.2.3", "01.2.3", "1.2.3; echo unsafe"]) {
    await assert.rejects(prepareRelease(version, directory), /plain SemVer/);
  }
  assert.equal(await readFile(join(directory, "Cargo.toml"), "utf8"), manifest);
  assert.equal(await readFile(join(directory, "Cargo.lock"), "utf8"), lockfile);
});
