# LunaFlux operations

## Current command boundary

The operator-facing commands are:

~~~text
lunaflux run ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
lunaflux doctor ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
lunaflux plan ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
lunaflux bench ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
lunaflux inspect-kernels ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
lunaflux legacy-doctor
lunaflux legacy-doctor --json
lunaflux legacy-plan MODEL
lunaflux legacy-inspect-kernels MODEL
lunaflux legacy-run MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256
lunaflux legacy-doctor MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256
lunaflux legacy-plan MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256
lunaflux legacy-bench MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256
lunaflux legacy-inspect-kernels MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256
~~~

All five canonical commands take one canonical absolute deployment-root label
plus an independent digest of its fixed `lunaflux.launch.json` descendant.
They share the recipe-specific runtime-instance admission that authenticates
the launch, model, kernel, policy, descriptor, tokenizer, bootstrap, and worker
executable joins. An unpinned `MODEL`, malformed suffix, wrong-case digest,
extra operand, or missing operand is rejected; it never falls through to
model-root preflight.

`run` is the production activation boundary. After the common admission it
verifies the assigned CUDA target and executable before creating the existing
worker, service, and listener owners. It publishes readiness only while that
exact listener and lower service remain live, and drain or failure clears
readiness. `bench` uses the same admitted deployment and live runtime owner for
its bounded local trial.

`doctor`, `plan`, and `inspect-kernels` stop immediately after the common
semantic join. They close every filesystem root before printing canonical
root-free evidence, and they do not probe or open a device, spawn a child, bind
a listener, compile, or JIT. Their successful terminal claim means only that
the authenticated release diagnostic, plan, or kernel join is available;
readiness remains false.

Every older model-root or separate-root spelling is available only through the
visibly named `legacy-*` compatibility namespace. `legacy-plan MODEL` and
`legacy-inspect-kernels MODEL` admit and close the model root, then stop at
`NotPrepared` because no independently supplied descriptor digest exists.
`legacy-doctor` is the bounded host-capability report. These compatibility
commands never activate a service and always report false readiness.

The legacy separate-root forms independently admit `MODEL_ROOT` and `KERNEL_ROOT`, bind
both canonical labels to their exact pinned capabilities, and authenticate
`DESCRIPTOR_REL` with the deployment-supplied lowercase SHA-256. The bounded
descriptor reconstructs the existing typed model metadata, paged model plan,
weight-file inspection, production paged-execution manifest and AOT artifacts,
bootstrap manifest/source/startup contract, and inert `DeviceWorkerPlan`.
Both roots close before CUDA inventory is checked. The admitted host-side plan
and the typed memory/capacity report remain available when CUDA is unavailable,
the assigned ordinal is absent, or its SM/BF16/cuBLASLt target does not match.
The separate-root `legacy-doctor` performs this existing physical preflight; the report
itself performs no probe and grants no authority. Those physical outcomes still
prevent activation. A complete match returns `LunaModelPreflightComplete` and
readiness false in these legacy five-argument preflight commands; their
`legacy-run` spelling does not start a listener, and `legacy-bench` does not invent benchmark
evidence. Production activation uses only the digest-suffixed one-argument
`run` form above.

This command surface sits on a strict byte-bounded configuration document
reader, focused immutable records, deterministic startup-capacity decisions,
and inspection-safe kernel identities. Native and OpenAI request admission also
share the canonical temperature/top-k/top-p/seed/stop vocabulary. These are
operational foundations, not proof that MODEL can be served.

The descriptor-backed report separates materialized and reserved weight bytes,
the activation reserved high-water mark, its workspace subregion, the single
activation/workspace arena, KV logical-page and reserved-stride geometry, and
kernel module file bytes. Its checked device total is exactly the weight arena
plus the activation/workspace arena plus the KV arena; workspace and artifact
file bytes are not added again. Descriptor v1 admits neither scheduler/cache
configuration nor service configuration, so those capacities are explicitly
unavailable. The remaining startup prerequisite after complete pinned
admission and a matching device is production service activation through the
live device-worker owner. A compiler/JIT-free OCI source contract, exact
build-context verifier, Linux-only build wrapper, and deployment runbook now
exist in `docs/DEPLOYMENT.md`. They were statically checked on macOS; no Linux
image, final-rootfs scan, SBOM, CUDA execution, or reproducible final image
digest has been produced or proven here.

The older explicit `lunaflux legacy-config-plan ROOT CONFIG_REL ...` form remains available
for authenticated model-config and deterministic KV-capacity explanation.
`lunaflux reference ...` remains an offline correctness command and is never a
production fallback.

## Kernel trust and inspection

Kernel configuration contains only a strict relative production execution-
manifest locator, an exact lowercase SHA-256 digest, and an AOT-only policy.
The independently pinned descriptor and immutable mounts are deployment trust
inputs; LunaFlux does not verify deployment signatures itself. The policy
string is schema and cannot grant verification authority.

The separate-root `legacy-inspect-kernels` summarizes only the existing production
`PagedExecutionAdmission`: immutable model/target/catalog/profile, operation,
module, entry-point, workspace, and KV geometry. It does not manufacture or
conflate a Phase-5 LunaTile capability-manifest admission. Inspection never
grants execution authority and cannot accept source, invoke a compiler, load a
device, or JIT.

The Phase 5 path emits deterministic CUDA source and recipes for the complete
dense-Llama BF16 graph families from an authenticated model plan, compiles the
candidate set through a deterministic two-build offline wrapper, and strictly
binds actual CUBIN identities to final launch contracts. The inert
`luna_kernel_bundle` join proves full plan coverage and produces canonical
manifest bytes plus the exact content-addressed module inventory. Runtime
admission still treats those bytes as untrusted claims and rederives the
complete semantic contract before module loading.

The full paged executor and spawned worker/service path are wired for BF16 and
the typed symmetric-I8 weight-only route. Ordered eager execution is the
production-worker default; startup-only graph capture exists only when an
explicit policy selects it. Exact-current-source physical evidence covers the
authenticated tiny BF16 model, its AOT graph and same-page KV state, the
spawned-parent boundary, one native listener request, bounded slow/fast-client
progress, ten sanitizer/ABI gates, and exact post-run GPU resource balance.
This does not claim positive I8, positive tensor parallel/NCCL, broad contexts,
cached-prefix execution, an approved OCI image, or benchmark promotion.

## Health and readiness

Health means the process can answer bounded diagnostics. Readiness is stricter:
the exact model, weights, kernel artifacts, worker startup contract, device
context, and executor must all be live. A successful CUDA inventory probe is
not readiness. A `NotPrepared` or `PhysicalGate` outcome always means
readiness is false.

Instance metrics and structured diagnostics use fixed vocabularies and bounded
cardinality. Raw prompts, generated text, filesystem paths, vendor messages,
device pointers, and request IDs are not metric labels or default log fields.

## Install and upgrade

1. Follow `docs/DEPLOYMENT.md`; the mandatory Linux build wrapper verifies the
   exact clean context and independently inventoried digest-pinned base before
   buildx. Never invoke a mutable base tag or treat an OCI label as authority.
2. Install one final digest-pinned LunaFlux image or executable and its exact
   AOT kernel artifact set. The final image still requires provenance, SBOM,
   and complete-rootfs scan evidence from the deployment environment.
3. Mount the model root read-only and keep the baked kernel root and container
   root filesystem read-only. The model mount marker is diagnostic only.
4. Run digest-pinned `lunaflux doctor DEPLOYMENT#sha256=...` and inspect its
   authenticated, inert release evidence.
5. Run the same digest-pinned `plan` and correctness gates before routing traffic.
6. Treat a new configuration schema, runtime image, model digest, or kernel
   manifest digest as a new immutable rollout candidate.

Do not replace files inside a live approved root. Stage a new immutable root
and start a new instance so all identities are re-admitted together.

## Drain, rollback, and shutdown

Stop new admissions first, retain bounded existing work, and wait for terminal
event acknowledgement before closing the worker. The instance-level drain
owner is the healthy shutdown path; process termination is not a substitute.

Rollback selects the prior complete runtime image, model root, kernel root,
configuration, and external approval receipt as one digest-pinned unit. Never
mix a prior binary with a newer manifest or mutate the active mount in place.
Never roll back through a mutable image tag.

## Troubleshooting

- `unsupported platform`, missing driver library, incomplete driver ABI, or
  failed driver initialization: correct the host runtime before model work.
- unresolved `pthread_*`, implicit GNU `posix_spawn` declarations, or a host
  older than glibc 2.34: reject the native build/host; do not patch the final
  linker command or weaken process descriptor closure in production.
- `no devices` or inventory failure: verify the process-visible assignment and
  container device policy.
- `not prepared: independently pinned runtime descriptor was not supplied`:
  use an explicit form with separate roots, descriptor locator, and the
  deployment-approved descriptor SHA-256.
- `preflight complete: production activation was not attempted`: the
  `legacy-*` five-argument diagnostic admitted all inert descriptor evidence and matched
  the probed target. It intentionally starts no live serving owner. Use the
  digest-suffixed one-argument `run` form for activation.
- `assigned device missing` or `assigned device target mismatch`: reject the
  rollout; the descriptor cannot select a different visible device or silently
  fall back.
- digest, identity, target, layout, entry-point, workspace, or eager-fallback
  mismatch: reject the rollout. Do not silently select another artifact.
- worker loss after readiness: fail affected requests, invalidate the device
  generation, and restart only through the retained exact startup contract.

Physical-CUDA numerical, sanitizer, leak, soak, and benchmark evidence remains
a release gate. The finite 10,000-request balance gate and the corrected v2
low-rate 24-hour host soak have passed. The separately frozen rerun1 binary was
still the rejected v1 harness and its exit 134 is a diagnosed harness false
negative, not promotion evidence. Host-only or fake-device results cannot
promote physical-CUDA readiness.
