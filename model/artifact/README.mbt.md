# Phase 1 reference artifact admission

`model/artifact` admits the exact bytes used by LunaFlux's offline Phase 1
correctness path. The caller supplies independently approved SHA-256 identities
for `config.json`, `tokenizer.json`, and one `safetensors` file. Their strict
relative locators are admitted once into an opaque `ArtifactSource`; loading
receives one separately caller-owned `ApprovedRoot`. The package never scans a
directory, opens an absolute path, executes metadata, or infers files.

`load_reference_bundle` is a synchronous whole-file host snapshot. Each strict
relative descendant is opened with component-wise no-follow traversal and a
final regular-file check. A same-handle stamp preserves per-file-before-total
limit precedence before the native immutable-snapshot transaction performs its
own stamp/read/trailing-probe/stamp checks. The file is deterministically closed
before bytes can be hashed, parsed, or published. A close failure wins over a
successful body; an earlier snapshot failure stays primary while close is still
attempted. The caller's root remains open and caller-owned.

The package has no production dependency on ambient or asynchronous filesystem
APIs. Namespace replacement cannot redirect an opened root or file. Concurrent
truncation, growth, or same-handle size/mtime/ctime change fails without
publishing a snapshot. Relative labels are locators, never trust identities;
the independently approved digests remain authoritative.

The bundle retains only the verified model `ContentDigest`; it cannot publish a
plan digest before an execution graph exists. A downstream Llama builder passes
that content plus complete graph and numeric semantics to `ModelPlan`, which
mints the exact execution identity. The tokenizer and configuration-file
digests remain separately available through `ArtifactIdentity`. Public errors
contain only bounded categories and never retain paths or file contents.

This package is not the production weight loader. Production loading must use
streaming or mapping and place tensor bytes directly into their final device
allocations through the materialization boundary. It must not retain a complete
model-sized host copy.

The `lunaflux reference` deployment boundary is the sole production caller. It
opens the operator-supplied absolute root, constructs the opaque relative
source, loads the bundle, and closes the root before reference execution or
console output.
