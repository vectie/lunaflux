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
Minimum observed host availability is 18,952,668 KiB (18.07 GiB), maximum GPU
use 96,489 MiB, and owned service swap zero.
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

The native all-reduce launch has 32 CTAs × 640 threads, 96 registers/thread
and 61,760 dynamic shared bytes. The retained reference includes that same
geometry/resource footprint (and some 19-CTA reductions). Different symbol
spelling is therefore insufficient to assert either a protocol difference or
an occupancy/resource cause. Actual protocol/progress still needs independent
evidence before changing the device submission policy.

The external paired analyzer now also reports a launch lead only for a unique
runtime correlation; ambiguous/missing correlations remain null. Evidence:
`/tmp/lunaflux-glm-dense-pair-launch-lead-20261020-v6.json`. No graph construction
or host-API total is substituted for a measured exposed dependency.

No production IR/scheduling change has been justified by these captures.
It would be incorrect to attribute the smaller spans to a compiler fix, or
to claim the historical spikes' cause is proved.

## Fresh paired reference: failed memory qualification

`/tmp/lunaflux-reference-bench-glm-dense-pair-20261020-v1` was a fresh vLLM
attempt with both ranks profiled, context 8,192, batch tokens 64, maximum
sequences/capture size 8, KV budget 256 MiB and both Docker containers limited
to 96 GiB with equal memory/swap limits. The single request was to preserve
the literal prompt, with only the model alias and streaming encoding changed.
No request latency/throughput benchmark was planned or run.

Checkpoint loading took 597.46 seconds. The desktop readiness loop was
temporarily paused while its **independent memory watcher and remote owners
continued running**, to avoid restarting this load because of the first
controller's ten-minute readiness budget. The checked-in controller now allows
30 minutes for bounded startup and waits for the capture-stop RPC before
retirement. Neither change affects model execution or measured GPU intervals.

During post-load warmup the watcher observes **15,631,756 KiB (14.91 GiB)**
available host memory on rank 0, below the mandatory 16 GiB reserve, at
**94,557 MiB** GPU use. It stops both owned containers. This is a **failed
memory qualification**, including a sampled reserve violation, not a passing
bounded run. Both GPUs subsequently show no owners and over 117 GiB host
memory available. No profiler-start/request trace was collected, so the table
above remains explicitly a comparison with the retained reference. The failure
and terminal journals are preserved; no reference result is invented.

The installed reference also warns that `w1_weight_scale_2` must match
`w3_weight_scale_2` and accuracy may be affected. That warning remains in the
logs; a reference engine is not a correctness oracle.

The configured GPU ceiling was not exceeded. The reserve guard is reactive;
these failed startup samples show why its stop action is not proof that every
instant stayed above the reserve. No additional paired-reference attempt or
memory-limit relaxation is made in this bounded campaign.

## Restoration and validation

At the start of this turn, the existing GLM service was already stopped:
rank 1 recorded an ABRT at 02:44, not a systemd OOM result, with zero swap;
the paired controller stopped rank 0. The installed artifact is `eb152338`.
The failure journal is preserved under
`/tmp/lunaflux-glm-dense-restoration-20261020-v1/before-rank{0,1}.txt`.

Both qualified `eb152338` production units were restarted through the existing
LunaNexa paired controller after all reference owners retired. Backend health
returns `healthy`; the complete `17 multiplied by 19` request returns **323**,
`finish_reason=stop`, 26 input / 6 output tokens with a 128-token budget. This
is restoration correctness, not a timing campaign or a fresh authenticated
customer-route acceptance test. Production still has 16 admission lanes; the
eight-lane profiling manifest and interposer are not installed there.

Observed restoration minima: rank 0 **18,817,576 KiB (17.95 GiB)** host
availability; rank 1 **20,582,752 KiB (19.63 GiB)**. GPU maxima are
**96,451 / 96,480 MiB**; both units report success, active, zero owned swap.
The temporary observer exits after the answer check; the production ranks
remain running. Evidence: `memory.txt` and `answer.request.json.reply` under
the restoration root. The earlier ABRT is retained, not retroactively called
an OOM or fixed by this investigation.

## What is fixed, and what remains

Fixed: missing ingress-child GPU observability, misclassification of a requested
Nsight-wrapper shutdown, and the lack of a two-rank causal/launch-lead report.
These are external diagnostic repairs. The eight new/affected script tests
pass (launcher 5, preparation 1, analyzer 2); the reference-owner regressions
also pass 9/9. The reference-window and probe scripts compile but contain no
unit tests. The actual approved-spawn child GPU fixture passed on Spark.
No fresh full native-suite or production-kernel sanitizer claim is made for
this tooling-only change.

**No production TP performance fix is qualified.** Model numerical contracts,
the pure `OrderedSubmissionPlan`, typed communication effects, CUDA/NCCL
lowering and serving selection remain unchanged. The code still submits
consumer work after NCCL submission, not after every collective completes.
The observed multi-millisecond queue lead confirms submission ahead of GPU
execution at the selected early sites; it does not prove optimal submission
in every workload.

The next causal boundary is collective progress after both ranks have entered,
separate from demonstrated early-arrival skew at other sites. A fresh paired
reference first needs a qualified startup memory envelope; any reduction of
startup-only graph capture must verify that the selected 64-row prefill path
is unchanged. Do not rerun the failed envelope, force a protocol from kernel
names, or promote a scheduling change merely because one trace has shorter
waits. The independently measured dense producer gap remains a different
optimization target, not an explanation of the historical TP spikes.
