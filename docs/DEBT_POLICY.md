# LunaFlux technical-debt policy

This policy prevents the compatibility and central-object growth observed in
mature inference engines from becoming LunaFlux's starting architecture.

## Structural budgets

- Prefer cohesive files below 500 lines.
- A file above 800 lines requires an architecture decision and a split plan.
- A package owns one stable responsibility and its public concrete types.
- No Scheduler, Config, ModelRunner, or Request object may become a universal
  dependency.
- No mixin-based feature assembly.
- No global mutable runtime context.

Line limits are review alarms, not an excuse for meaningless file splitting.
Dependency direction and coherent state ownership remain primary.

## Feature placement

- API compatibility belongs in api adapters.
- Model-specific logic belongs in model-plan builders.
- Scheduling policy belongs in scheduler.
- Physical KV ownership belongs in kv.
- Prefix matching belongs in prefix.
- Hardware selection belongs in device and kernels.
- CUDA declarations belong only in internal/cuda.

A feature that requires conditionals in three of these layers must first define
a typed capability and an architecture decision.

## Configuration

- Components receive narrow immutable records.
- There is no engine-wide configuration object passed to every constructor.
- Environment variables are bootstrap inputs only.
- Unknown configuration is rejected.
- Defaults are versioned and printed in the resolved startup plan.
- Deprecations last one minor release unless explicitly committed otherwise.

## Hot path

After warm-up, a token step must avoid:

- general heap allocation;
- string construction and parsing;
- hash-map growth;
- model-family reflection;
- dynamic kernel compilation;
- unbounded queues;
- blocking network or filesystem operations.

Preallocated arrays, views, arenas, integer capability IDs, and bounded rings
are preferred. Performance-sensitive unsafe/native code requires a safe wrapper
and a differential test.

## Hardening and validation placement

- Consume deployment-staged artifact identity and retain typed runtime state.
  Checkpoint inspection reads bounded headers; loading reads requested tensor
  ranges. Neither performs complete-weight hashing or repeated authentication
  scans. Optional integrity verification belongs to acquisition tooling, not
  engine startup or inference.
- Production token execution must not perform cryptography, filesystem
  validation, evidence rendering, diagnostic host/device transfers, canary
  observation, or qualification-only scans.
- Runtime packages consume admitted model, kernel, and device contracts. They
  must not depend on release-evidence or campaign packages; physical evidence
  decides promotion in the release pipeline, not dispatch inside the engine.
- Ingress keeps only the bounded parsing, credential, and request checks needed
  for untrusted input. It must not replay artifact or deployment admission.
- The normal edit loop is formatting, warning-denied native checking, and tests
  for affected packages. Sanitizers, physical hardware campaigns, soaks,
  benchmarks, and release assembly run only for their changed boundary or a
  phase/release gate.

Remove hardening that increases inference latency, startup payload traffic, or
routine edit/test latency. Keep bounds, dtype/shape validation, cancellation,
and deterministic resource ownership: these are execution correctness, not
optional security work. Pure compiler/cache identity generation is not a
checkpoint authentication pass.

Checkpoint build, resume and download helpers follow the same policy: do not
scan source archives, binaries or CUBINs for checksums. Use command success and
compiler terminal status, and keep new output paths non-overwriting. Historical
receipts remain historical; these helpers do not claim content authentication.
Compatibility entry points must launch the same implementation, not preserve a
hidden shell fallback that rescans payloads or compiles AOT modules twice for
bytewise comparison. Materialization must not repeat runtime preparation or
scan copied model/kernel payloads to create a full checksum inventory. Small
newly generated plan/descriptor identities are computed once offline; existing
artifact labels are consumed, not reauthenticated.

MiniMax host materialization likewise consumes the immutable manifest's
construction-time label without reserializing its tensor/layout entries or
rehashing them on load. The label comparison is a plan association, not a
payload authentication claim. Allocation budgets, ranges, shapes and dtypes
remain execution-correctness constraints.

DeepSeek host materialization and all segmented device upload modes likewise
consume their immutable construction-time plans without replaying numeric
manifest validation or canonical layout hashing. Qwen candidate export/release
binding parses config semantics without comparing a config payload checksum.
MiniMax checkpoint startup also consumes a supplied identity label instead of
hashing config bytes. `ROOT#label=HEX`, `LUNA_MODEL_CONTENT_LABEL`, or the first
label in the selected component inventory supplies it. Only a 72-byte inventory
prefix is read for that association; there is no config-hashing fallback and no
claim that this label verifies tensor or configuration contents.

The shared tokenizer-file loader and reference bundle follow this policy too:
parse the bytes needed for execution without a separate config/tokenizer hash
pass. Digest-shaped compatibility fields retain caller-supplied labels; they
do not certify the loaded bytes. Bounds and semantic parsing remain required.

Checkpoint consumers that only require model execution semantics must use
spec-only parsing, without computing and discarding a configuration digest.
GLM Flash and DeepSeek checkpoint startup follow this route. Do not replace
discarded hashes with fabricated content identities.

Weight conversion must not reopen an entire converted payload to hash it or
repeat source inspection after copying. Compatibility `*_sha256` fields may
carry supplied or plan-derived identity labels; report their kind and never
describe such labels as verified payload checksums. Tuning inputs and competing
module choices are parsed for execution compatibility, not reauthenticated by
opening and hashing every alternative. Fresh CUBIN packaging does not require
a second checksum inventory traversal.

BF16 release binding also consumes declared toolchain, receipt and module
labels without payload rehashing. The kernel producer and shared bundle join
must not hash each CUBIN again or compare entire first/second-build copies.
They retain label association, source/recipe compatibility, module-size budgets
and exact launch/operand/workspace checks. Optional tuning and fold snapshots
are parsed for execution scope, not authenticated by their CLI labels. Legacy
deterministic-receipt names do not constitute a new proof of payload equality.

Qwen tied-output inspection follows the declared model semantics: reference
zero owns both embedding and output-head storage. A redundant physical head is
checked for shape/dtype/range compatibility but its bytes are ignored, never
scanned for equality with the embedding. Metadata-only and file inspection use
the same binding rule. Source-manifest metadata reports the ignored copy, not
verified byte equality; the legacy comparison-buffer constructor argument does
not allocate or retain buffers.

All worker bootstrap receivers consume the supplied footer label without
hashing, re-encoding or byte-comparing frames, including the direct I8 entry
point. They publish parsed typed records with one owned copy. Magic/version,
length, UTF-8, target, KV geometry and allocation ceilings still validate before
ownership is published. The label associates startup records; it does not
authenticate transport bytes. Construction-time control-plan identity remains
separate from receiving and reauthenticating an existing payload.

## Compatibility discipline

- One implementation of the engine is authoritative.
- Experiments live behind an explicit package/capability boundary and are
  removed or promoted before the next phase ends.
- Do not retain parallel V0/V1 engines indefinitely.
- Do not accept arbitrary Python extensions for compatibility.
- Unsupported model variants produce typed incompatibility reports.
- Silent slow-path or precision fallback is forbidden.

## Resource ownership

- GPU and native resources have explicit close operations.
- Close order is documented and tested.
- Finalizers, when used defensively, are not the primary lifecycle mechanism.
- Every cancellation and failure test asserts page, request-slot, and worker
  resource balance.
- Stale IDs use generation checks rather than pointer identity.

## Testing debt

No feature is complete without:

- public black-box behavior tests;
- illegal state-transition tests;
- deterministic scheduler fixtures when scheduling changes;
- cancellation and exhaustion coverage;
- compatibility failure fixtures;
- benchmark comparison when a hot path changes.

The aggregate repository boundary also runs the native MoonBit compiler with
warning 73 enabled and warnings denied. Redundant package annotations therefore
remain a checked debt boundary instead of relying on periodic manual cleanup.

Snapshots are reviewed evidence, not blindly updated output.

## Removal rule

Temporary code must name:

- the issue or phase that owns it;
- the condition under which it is removed;
- the latest phase by which removal occurs.

TODO without an owner or removal condition is rejected. HACK and FIXME are
release-gate failures in hot-path and native-ABI packages.

## Review checklist

Every change should answer:

1. Which package owns the new state?
2. Is the dependency direction preserved?
3. Can an unsupported combination fail at startup?
4. Does steady-state work allocate or block?
5. Is native ownership explicit?
6. Are public errors bounded and payload-safe?
7. What deterministic test proves the transition?
8. What benchmark would expose a regression?
