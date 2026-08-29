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

- Authenticate deployment, artifact, and live-device identity once during
  startup admission; retain the resulting typed runtime state.
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

Hardening that materially increases steady-state latency, memory traffic, or
routine edit/test latency requires an architecture decision and a measured
justification. Prefer moving it to startup or an explicit qualification mode.

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
