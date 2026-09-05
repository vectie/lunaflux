# Nonblocking control progress

The Qwen decode profile identified a control listener's 1 ms accept/read/write
poll in the serialized inference owner's progress chain. GPU backpressure
caused a control turn on each retry, making an unrelated idle HTTP socket delay
worker completion retirement and the next token submission.

Control turns now use the existing reactor's zero-timeout I/O operation. Ready
accepts, reads and writes still progress; an empty poll returns to inference.
The existing bounded control fairness policy and singular lifecycle owner are
unchanged. No new task, native ABI, scheduler or model-family branch is added.

Polling and connection lifetime are separate policies. Each accepted connection
starts a monotonic input-idle window; received bytes refresh that window. A
response starts its bounded write window. Empty polls retain partial input and
output until the corresponding configured deadline expires. Clock failure or
reversal retires the peer. Drain and cancellation retain deterministic cleanup.
The deadline decision is a pure scalar function; clock/socket effects remain
in the transport owner, outside the compiler and scheduler.

Focused regressions cover zero-wait polling, deadline boundaries, fragmented
requests across delays greater than 1 ms, idle-client retirement, health,
readiness, malformed requests and drain. This is the production fix, not the
earlier diagnostic that disabled control progress entirely. Physical benchmark
results from the commit-pinned Linux rerun follow.

## Real Qwen A/B result

The committed fix is `37cb1ad`. A clean committed-source Linux release build
was used for the runtime executable; no dirty-worktree changes or diagnostic
control-disable/profiler patches were included. The existing model, worker,
bridge, deployment and compiler-generated GPU kernels were reused unchanged.
The new runtime SHA-256 is
`17a5c53bc610e9e7288603691ebacca1c4bc8f6a8c3b4187f9229bac5c02b2b8`;
the old runtime is
`f5b838d3eba5feff48c5e854e05c80261fd53f736bc7bda205d1b035b9ea3f03`.

Qwen3-0.6B BF16, RTX 5060 Ti, token vector **[input=59, output=256,
concurrency=1]**, greedy, ignore EOS, loopback token-ID SSE. Each version ran
one excluded warmup and three measured requests. The fixed version ran first,
then drained; the old version was started and remeasured on the same idle GPU.
No profiler or concurrent control traffic ran during these timed requests.
Numbers are output tokens divided by client `curl time_total`, including
prefill and transport, not isolated decode ITL or TTFT.

| Runtime | Trial 1 tok/s | Trial 2 tok/s | Trial 3 tok/s | Median tok/s |
| --- | ---: | ---: | ---: | ---: |
| Old normal control polling | 118.306 | 118.444 | 118.684 | **118.444** |
| Fixed control polling (`37cb1ad`) | 182.754 | 183.062 | 182.772 | **182.772** |

Throughput improves **54.31%**. Median request time falls from **2.161361 s**
to **1.400654 s**, saving **0.760707 s/request** or **2.9715 ms/output token**
amortized. All **768/768** generated token IDs match the old version across
the three trials. This is a general transport/effect-scheduling improvement,
not a Qwen-specific rule or a new tile compiler kernel optimization.

A separate live-control regression generated another 256 tokens while issuing
12 requests each to `/healthz`, `/readyz` and `/metrics`: **36/36 returned
HTTP 200**, with maximum observed client latency **2.503 ms**. This run is
excluded from the throughput table. The Linux real-worker operational HTTP
campaign passed, including 2,048 empty polls under a 500 ms guard, fragmented
input across a 5 ms pause, actual idle timeout, malformed requests, health,
readiness and drain. The 54 focused socket tests also passed on Linux; local
socket/runtime/HTTP tests passed 170/170. The dependency C build still emits
the pre-existing `posix_spawn_file_actions_addchdir_np` declaration warning.

Both A/B instances acknowledged drain, exited with code zero, closed their
children and emitted empty stderr. No GPU compute process remained. No
production instance or deployment was changed.

For context only, the preceding same-vector baseline run measured vLLM 0.24.0
at 281.073 tok/s and SGLang 0.5.2 at 267.366 tok/s. Relative to the fixed
runtime those are **1.538×** and **1.463×**, respectively. Those two frameworks
were not rerun in this control-fix A/B. Decode attention's cross-workgroup KV
partition/merge optimization remains unimplemented in this fix; the earlier
[kernel profile](FUNCTIONAL_COMPILER_GAP_PROFILE_2026-09-05.md) still motivates
that next step. Three short trials on this one vector do not establish a full
length/concurrency matrix, tail latency or broader model-quality parity.

## Reproduction records

- Committed-source archive SHA-256:
  `755af12045f76b712a145ad279bf2a550c8bbf3b84f699155715da2c0d7e3891`.
- Remote source: `/dev/shm/lunaflux-control-source-37cb1ad`.
- Raw fixed run: `/dev/shm/lunaflux-control-fixed-37cb1ad-20260905-r1-lunaflux`.
- Raw old run: `/dev/shm/lunaflux-control-baseline-20260905-r1-lunaflux`.
- Archive: `/dev/shm/lunaflux-control-results-37cb1ad-20260905-r1/results.tar.gz`.
- Download: `/private/tmp/lunaflux-control-results-37cb1ad-20260905-r1.tar.gz`.
- Archive SHA-256:
  `522cfd818ae000464f8247a48a11983d27bddca2e550af5c0c3fa6ba90f518de`.

The archive includes both request bodies, exact launch arguments and binary
identities, raw SSE/timestamps, control responses, lifecycle logs, the passing
operational regression and the MoonBit orchestration scripts. Historical
`profiled-*` capture filenames were retained by the reused driver; both runs
explicitly used `plain` mode and had no profiler attached.
