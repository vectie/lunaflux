# Approved tiny-model spawned physical execution campaign

This native-only harness authenticates a fully materialized LunaFlux deployment
through `preflight_release`, requires the exact upstream
`stas/tiny-random-llama-2` BF16 model, upstream tokenizer, canonical single-row
paged plan, and an independently supplied child-executable SHA-256, then calls
the production one-argument `runtime_instance.prepare` path.

Success means the real spawned child sent `Ready` only after constructing its
physical executor and the operator owner retained exactly one pre-listener
service preparation. A narrow operator-owned offline validator consumes that
preparation without exposing the service, worker, scheduler, device, or CUDA
owner. It sends one canonical framed token request through the existing Luna
service boundary, checks the pinned corpus tokens `1031,2185`, proves two
consecutive one-row/one-token plans (final prefill then same-page decode), and
proves the same single-page geometry across both token events. Scheduler
telemetry may expose either that one live page or the fully retired request
with the original free-page balance because semantic-event consumption can lag
request retirement; both states are checked exactly. It then disconnects the
stream and deterministically drains, closes, and reaps the child before
publishing evidence.

The request advertises the model's full 256-token context ceiling. Production
startup and scheduler admission therefore require an 8-token-page release to
provide 32 physical pages, 32 block-table entries per request, at least 32
worker plan-page cells, a worker/device sequence ceiling of at least 256, and
the scheduler's independently validated positive emergency decode-page
reserve. The two-token numerical observation requires one page; the larger
geometry is startup capacity, not a claim that the probe fills all pages.

This is a token-level numerical execution result for the one checked-in tiny
BF16 model case. It does not bind a listener or establish traffic readiness,
does not observe selected logits, and makes no serving, performance, tensor
parallel, or I8 claim.

Run on an independently provisioned target host only after creating an approved,
digest-pinned deployment namespace containing the model, tokenizer, runtime
descriptor, instance policy, AOT kernel manifest and modules, and the canonical
device-worker child:

```sh
moon run --target native tests/approved_model_spawned_physical -- \
  ABSOLUTE_DEPLOYMENT_ROOT#sha256=LOWERCASE_LAUNCH_SHA256 \
  LOWERCASE_CANONICAL_CHILD_SHA256
```

The repository's narrow offline approved-model binder and materializer can
create this exact launch namespace from independently authenticated sm120
compiler output and a digest-pinned child executable. They do not compile or
open a device. Missing target artifacts, a non-matching model/tokenizer/plan/
child identity, substituted expected token, unavailable CUDA, child bootstrap/
execution failure, nonconsecutive plans, KV-page drift, or incomplete cleanup
all fail closed without printing success evidence.

## Bounded native-listener readiness campaign

The same pinned deployment can additionally exercise the production
`runtime_instance` listener owner. This operator-only path binds the existing
native pipeline listener on loopback, observes health and traffic readiness as
separate states, sends exactly one authenticated greedy two-token request, and
requires the canonical `Accepted`, `Token`, `Token`, `Usage`, `Completed`
event order with tokens `1031,2185`. It then closes the client, proves exact
request, network, and KV balance, retires the listener, drains the service, and
reaps the spawned child before returning opaque proof scalars.

```sh
moon run --target native tests/approved_model_spawned_physical -- \
  serving \
  ABSOLUTE_DEPLOYMENT_ROOT#sha256=LOWERCASE_LAUNCH_SHA256 \
  LOWERCASE_CANONICAL_CHILD_SHA256
```

This establishes only a bounded native TCP serving-readiness slice for the
pinned model and request. It does not validate TLS, public-network exposure,
concurrent clients, throughput, latency, or performance.

## Embedded device-greedy qualification

The `device-greedy` mode compares two independently admitted spawned-child
routes for the same two-token request. The first descriptor selects the actual
production `embedded_cuda_greedy_v1` full-decode reducer. The second uses the
production full-logits host sampler as an independent referee. Both cross
`descriptor_file -> worker_wire -> child bootstrap -> PagedDeviceExecutor`;
the result exposes only root-free counters after both child owners close.

The route records the embedded reducer's exact eight-byte
`token_i32,status_i32` readback for each plan, the authenticated graph path,
the lowest-token-id tie rule, and first-nonfinite-token status rule. The fixed
approved tiny model supplies two distinct physical logit rows, not injectable
tie or nonfinite rows. Tie/nonfinite behavior is therefore bounded to the AOT
source and host-contract tests; nonfinite model artifacts are rejected before
execution and are not mislabeled as physical reducer observations.

```sh
moon run --target native tests/approved_model_spawned_physical -- \
  device-greedy \
  ABSOLUTE_EMBEDDED_LAUNCH#sha256=HEX \
  ABSOLUTE_HOST_REFEREE_LAUNCH#sha256=HEX \
  LOWERCASE_CANONICAL_CHILD_SHA256
```

`device-greedy-fused-v2` additionally requires a descriptor-pinned canonical
fused production runtime. The materializer's
`fused-v2-device-greedy-v1` profile consumes an externally prepared aggregate
runtime plus the sealed lower production-V2 campaign's exact QKV, readonly
attention, residual, and identity pins. It re-admits the aggregate normally,
checks every module/source/symbol join, augments the exact kernel-root
inventory, and fails closed if authentic approval, runtime-device, or KV
fallback identities are absent. Neither mode grants manifest or promotion
authority.

The sealed sanitizer runner is
`scripts/run-spawned-device-greedy-physical-campaign.sh`. It runs memcheck,
racecheck, and initcheck across the parent and both spawned children, requires
identical observations and closed GPU/process resources, and publishes exact
`FILES.sha256`, `RESULT.txt`, and `OUTER_SEAL.sha256` transaction files.

The harness and its hostile/static validators are source-complete. No
`device-greedy` or `device-greedy-fused-v2` campaign has yet run on NVIDIA
hardware, so this README records software reachability only, not physical
correctness, sanitizer, resource-balance, or performance evidence.
