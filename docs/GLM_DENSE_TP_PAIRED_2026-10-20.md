# GLM first dense-FFN reductions: paired-rank investigation

## Scope and decision boundary

The request is to trace both GPU ranks, isolate the first two dense-FFN TP
waits, fix a demonstrated cause through the existing functional compiler,
and compare the execution path with vLLM. This is **not an overall benchmark**.
The selected engine remains `eb152338`: local-combined BF16 MoE, BF16
round-before-reduce projections, scales plus bulk expert transport. No model
payload hashes, numerical tolerance changes, or compiler/reference coupling
are introduced.

The retained rank-1 captures in `GLM_PROJECTION_TP_2026-10-19.md` recorded
dense layer 0/1 reductions at 3.788/3.226 ms and 4.529/5.447 ms. Those are real
recorded spans, but their attribution was unresolved. They cannot establish
network transfer time or a persistent compiler scheduling defect by themselves.

Budget: paired baseline, one single-rank capture control, and only a
demonstrated implementation correction. Failed setup/capture evidence remains
separate from performance evidence. Restore the existing qualified service.

## Repairing the missing measurement

The production ingress worker executes through a descriptor-based `fexecve`.
Its environment deliberately contains only deployment-time runtime settings;
Nsight's injection/session settings do not cross that boundary. Profiling the
HTTP parent therefore did not prove ingress **GPU** coverage.

`tests/process_native_fixture/nsys_exec_bridge.c` is an **offline LD_PRELOAD
interposer**, not linked into the runtime. It snapshots the observed profiler
settings before fork and forwards them at descriptor execution. It does not
change CUDA kernels, executable bytes, numerical laws, compiler selection,
runtime process code, or request-path behavior outside a profiling launch.
No unrelated environment variables or credentials are copied by the bridge.

`scripts/qualify-nsys-worker-trace.mbtx` compiles the exact approved-spawn
implementation with a tiny CUDA child. The child verifies inherited directory
descriptors, exclusion of an unrelated environment sentinel, CUDA execution,
and deterministic release. Actual child GPU memset activity is present in
the resulting SQLite, not just a parent process entry.

- `/tmp/lunaflux-nsys-worker-20261020-v1`: compile failure against CUDA 13's
  changed context-create declaration; no GPU test claim.
- `/tmp/lunaflux-nsys-worker-20261020-v2`: child GPU capture and release pass.

The desktop analyzer `scripts/analyze-glm-paired-tp.mbtx` matches collective
order and immediate same-stream producers. Wall clocks are **not** assumed
synchronized: observed timesync jitter was tens of milliseconds. For each
matched two-rank collective, each rank must start before its peer completes.
The intersection of these causal constraints bounds a constant inter-trace
offset. It does not assume equal completion timestamps, and rejects empty
intersections or mismatched operation/producer sequences. Clock drift within
the short capture and profiler perturbation remain limitations. Early-arrival
bounds are not a measurement of network-only transfer time.

## Paired captures and limits

The literal request is
`/tmp/lunaflux-glm-service-benchmark-policy-combined-20261019-v1-r1/words-7936-c1-lane0.request.json`:
7,973 input tokens, C1, maximum one output token, 64 physical rows per chunk,
TP2, NVFP4 checkpoint with its declared per-tensor representations, context
8,192, eight resident slots, no prefix/speculation. Short warmup precedes
collection. No overall latency/throughput sweep was run.

1. `/tmp/lunaflux-glm-http-dense-pair-20261020-v1` used unchanged 16 admission
   lanes and an 800 ms collection timer. The headroom guard stopped it; the
   sampled host reserve briefly fell below 16 GiB during shutdown. It is a
   **failed memory qualification**, not an accepted run. Both GPU traces were
   recovered after owners retired. Its complete first chunk shows dense 0/1
   spans of 0.230/0.173 ms on rank 0 and 0.164/0.205 ms on rank 1. This is a
   counterexample worth retaining, not a production speedup claim.
2. Setup roots v2–v5 retain rejected symlink/missing per-rank artifact attempts.
   The isolated diagnostic manifest now uses hard links to existing regular
   artifacts, the rank's own CUBIN plus sampling CUBIN, and rank-0-only HTTP
   credentials. There is no new compilation or checkpoint scan.
3. `/tmp/lunaflux-glm-http-dense-pair-20261020-v6` uses a 150 ms collection
   timer and eight waiting/admission lanes to reserve profiler host memory.
   **GPU geometry, resident slots, context, binary, CUBINs and C1 work remain
   unchanged.** This is not a production capacity reduction. The full literal
   HTTP request reaches `[DONE]`, with 7,973 input and one output token.

For v6, observed minimum host availability is **17,989,492 KiB (17.16 GiB)**;
maximum observed GPU allocation is **96,489 MiB**. Owned swap remains zero.
Both GPU owners retire before CPU-only SQLite export. Its controller nevertheless
reports failure: the memory watcher classified the deliberately terminated
Nsight wrapper's exit 143 as a crash after the complete response. The original
failure marker is preserved. The launcher watcher now distinguishes only an
explicitly requested, known-profile-wrapper TERM exit from unexpected exits;
ABRT and other failures remain errors. A regression test covers this distinction.
This is a diagnostic lifecycle repair, not a waived model correctness failure.

## Observed first-three dense execution chains

v6 records rank-0 kernels at 191,136,912–399,587,888 ns and rank-1 kernels at
143,488,784–351,053,136 ns. These raw clock origins are not comparable directly.
The common 64-collective prefix gives an offset interval width of 0.253728 ms.
The last incomplete collective/window is excluded from matching.

| First-chunk operation, ms | Native rank 0 | Native rank 1 | Retained vLLM rank 1 |
|---|---:|---:|---:|
| Layer 0 gate/up | 1.650368 | 1.709856 | 0.513536 + 0.004832 activation |
| Layer 0 down | 0.425248 | 0.430816 | 0.247392 |
| Layer 0 reduction | 0.226720 | 0.158272 | 0.239808 |
| Layer 1 gate/up | 1.668352 | 1.645536 | See retained detailed timeline |
| Layer 1 down | 0.409504 | 0.365824 | See retained detailed timeline |
| Layer 1 reduction | 0.158688 | 0.232832 | 0.174592 |
| Layer 2 reduction | 0.134208 | 0.168992 | 0.163200 |

The first two same-stream producer-to-reduction gaps are **3.968/5.056 µs**
on rank 0 and **2.144/3.520 µs** on rank 1. Rank-1 early-arrival upper bounds
are 0.060640/0.194368 ms; lower bounds are zero. These observations exclude
multi-millisecond peer-arrival waits **in this capture**, not in every run.

Per-rank chain, layer 0 (durations, not a synchronized cross-host drawing):

```text
rank 0: gate/up 1.650 ms → down 0.425 ms → gap 0.004 ms → reduce 0.227 ms
rank 1: gate/up 1.710 ms → down 0.431 ms → gap 0.002 ms → reduce 0.158 ms
vLLM r1: gate/up+activation 0.518 ms → down 0.247 ms → reduce 0.240 ms
```

No compute/reduce overlap is being claimed for these dependency chains.
The remaining visible layer-0 dense difference is primarily **producer
compute**, not the previously attributed 3–5 ms reduction spans. This does
not by itself identify an instruction-level cause or justify a new tile plan.

Source checks: our `dense_block.mbt` binds these first dense layers' BF16
weights and a shape-aware SwiGLU plan; `matrix.mbt` lowers a 16-row × 64-column
paired tile. The installed reference uses local matrix output followed by
`RowParallelLinear` reduction; `PyNcclCommunicator.all_reduce` submits on the
current stream. The generic `Kernel2` name alone does not establish its
precision, tile geometry, or complete vendor implementation. Do not infer
FP4 compute merely because the overall checkpoint is named NVFP4.

Reference timeline is retained, **not a fresh vLLM run this turn**:
`/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2/rank1.sqlite`.
The first 64-row chunk of the same literal long request is compared at the
operator boundaries; reference outer anchors and maximum output count differ.
No end-to-end ratio is calculated. Native symbol `RING` alone still does not
prove its NCCL protocol; reference `RING_LL` explicitly names LL.

Paired reports:
`/tmp/lunaflux-glm-dense-pair-aborted-20261020-v1.json`,
`/tmp/lunaflux-glm-dense-pair-first3-20261020-v6.json`, and
`/tmp/lunaflux-glm-dense-pair-prefix-20261020-v6.json`.

## Single-rank control and host/GPU distinction

`/tmp/lunaflux-glm-http-dense-egress-20261020-v1` completes the same literal
request and retires both owners with the corrected diagnostic watcher. It
profiles rank 1 only, retaining the eight-admission-lane diagnostic envelope.
The first dense reductions are **0.150784 / 0.542528 / 1.766240 ms**.
The preceding layer-1 recurrent reduction instead takes **6.484000 ms**.
Thus long spans can move between sites with unchanged kernels. This does
not establish a fixed dense-0/dense-1 defect, nor explain the historical spikes.

Host API sums are also not exposed critical-path time. The paired rank-0
capture contains 178.802 ms accumulated in `cudaStreamWaitEvent`, but its
first two dense NCCL launches have already returned **3.192608 / 6.864624 ms
before** their GPU kernels start. In the single-rank control, the long
recurrent-1 NCCL launch returns **3.530400 ms before** GPU start, and dense-1
returns **11.810448 ms before** GPU start. Those particular waits cannot be
attributed to late submission on the observed rank. The missing peer in that
control still prevents assigning the delay to peer arrival or transport.
Queue lead is not wasted time and must not be added to GPU duration.

The paired capture also contains a useful counterexample to a universal
late-peer explanation: recurrent layer 4's reduction lasts **3.442528 /
3.379264 ms** on ranks 0/1, while the causal rank-1 early-arrival upper bound
is only **0.067072 ms** (rank-0 upper bound 0.186656 ms). Both launches were
submitted more than 17 ms before GPU start. Most of this span is therefore
after both kernels have begun, not host enqueue starvation or a millisecond
peer-start skew. This does **not** distinguish NCCL device progress, proxy/
transport progress, or profiling disturbance. By contrast, recurrent layer
30 has a demonstrated rank-1 early-arrival lower bound of 0.386080 ms.
There are multiple kinds of delay; they must not all be called dense-FFN waits.

The external paired analyzer now also reports a launch lead only for a unique
runtime correlation; ambiguous/missing correlations remain null. Evidence:
`/tmp/lunaflux-glm-dense-pair-launch-lead-20261020-v6.json`. No graph construction
or host-API total is substituted for a measured exposed dependency.

No production IR/scheduling change has been justified by these captures.
It would be incorrect to attribute the smaller spans to a compiler fix, or
to claim the historical spikes' cause is proved. Fresh paired vLLM and service
restoration evidence follow when complete.

At the start of this turn, the existing GLM service was already stopped:
rank 1 recorded an ABRT at 02:44, not a systemd OOM result, with zero swap;
the paired controller stopped rank 0. The installed artifact is `eb152338`.
The failure journal is preserved under
`/tmp/lunaflux-glm-dense-restoration-20261020-v1/before-rank{0,1}.txt`.
