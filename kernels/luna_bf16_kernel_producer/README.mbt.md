# Complete BF16 offline kernel producer

This native-only package is the typed join between operation-neutral BF16
lowering candidates and exact outputs from the isolated offline compiler
driver. Actual CUBIN bytes establish content-addressed catalog families before
the full-graph launch contracts exist. Each family package then performs its
strict post-compile binding, and `luna_kernel_bundle` emits the schema-v2
execution manifest plus exact module inventory.

Equivalent same-shape operations share one release entry point and module;
their recipes, operands, operation IDs, and final contracts remain distinct.
Layer-sensitive rotary and attention catalog semantics retain their exact KV
layer even when their device code is reusable.

The package imports no filesystem, process, CUDA, compiler, PTX, JIT, or
runtime-service authority. The shell producer only creates offline compiler
outputs and receipts; those are not runtime-admitted until this typed join
succeeds.

The remote compiler boundary accepts no model root or ambient search path. A
caller first uses `prepare_luna_bf16_candidate_set` to derive every source,
symbol, operand, recipe, key, and order from one authenticated `ModelPlan`,
canonical KV layout, exact profile, target, and compiler policy. Model identity
and compiler policy retained in the opaque result are inert values, not model,
filesystem, compiler, process, device, or runtime authority. The dedicated
offline exporter writes `sources/<key>.cu`, `recipes/<key>.recipe`, the
canonical `candidate-set.v1`, and a sorted SHA-256 inventory outside that
candidate root. On the CUDA build host:

```sh
candidate_inventory_sha=$(sha256sum "$CANDIDATE_INVENTORY" | awk '{print $1}')
toolchain_manifest_sha=$(sha256sum "$TOOLCHAIN_MANIFEST" | awk '{print $1}')
nvcc=$(realpath /usr/local/cuda/bin/nvcc)
scripts/build-luna-bf16-kernel-set.sh \
  "$nvcc" \
  "$TOOLCHAIN_MANIFEST#sha256=$toolchain_manifest_sha" \
  "$CANDIDATE_ROOT" \
  "$CANDIDATE_INVENTORY#sha256=$candidate_inventory_sha" \
  "$NEW_COMPILED_SET"
scripts/verify-luna-bf16-kernel-set.sh "$NEW_COMPILED_SET"
```

The output contains only `compiled-set.v1`, its exact `files.sha256`, one
receipt per operation, and deduplicated `sha256/<digest>.cubin` files. The
compiled-set record is transport evidence, not catalog or runtime admission.
An offline caller reconstructs `LunaBf16OfflineCompilation` values from the
authenticated candidate objects, exact receipt fields, and referenced CUBIN
bytes; `produce_luna_bf16_kernel_release` performs the authoritative typed
join.

Production consumers of candidate source/recipe bytes are restricted to the
offline filesystem exporter. The approved-model physical campaign may inspect
the same opaque candidates to bind independently authenticated compiler
receipts. Runtime, service, scheduler, device, and worker packages may not
import this producer.

## Offline profiler evidence and promotion join

`collect_luna_bf16_profiler_evidence` admits a bounded, canonically ordered
set of raw offline observations. Every candidate operation must have a
contiguous declared shape set and the configured number of trials. Each shape
must win on summed integer nanoseconds, every observation must carry positive
correctness and dispatch-canary evidence, and the complete end-to-end trial set
must stay within the configured integer non-regression allowance. Shapes are
bounded by the exact candidate-set PagedV3 profile.

`promote_luna_bf16_kernel_release` then joins that evidence to the exact model,
target, compiler identity and flags, paged profile-priority capture, ordered
launch contracts, candidate source/recipe digests, uniform offline-driver
identity, compiled-operation toolchain identity, and schema-v2 release-manifest
digest. The result is opaque and root-free.

The collector does not run a profiler or referee; supplied timings,
correctness digests, and canaries remain raw claims. The opaque join is local
software evidence only. It is not physical benchmark reproduction, external
deployment approval, module-loading authority, execution authority, or serving
readiness. Runtime JIT remains absent.
