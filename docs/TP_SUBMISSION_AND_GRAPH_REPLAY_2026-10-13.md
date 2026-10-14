# TP submission and graph replay

## Question and bounded experiment

The existing decoder submits a collective, queues its consumers, then waits
for `cuStreamQuery` before submitting the next collective. This is a GPU
completion dependency in the host submission loop, not a necessary dependency
between same-stream operations. The previous scoped trace measured 132 NCCL
calls per GLM step and substantial pre-collective gaps; profiling overhead and
peer wait prevent interpreting those gaps as recoverable wall time.

Compare three configurations on the retained six literal prompts, one excluded
warmup, unchanged AOT/numerics/TP2 placement: per-site completion, queued
submission, then queued submission with captured compute segments. Budget two
implementation stages with focused failure/correctness tests and repeated
end-to-end cohorts. Keep vLLM results and comparison outside compiler policy.
No payload hashing. Both hosts retain a 16 GiB available-memory reserve and
each task/device ceiling remains at most 100 GiB, with no task swap.

## Functional separation

`parallel_ir` describes immutable ordered compute intervals and collective
cuts, submission policy, and replay policy. Same-stream ordering preserves
producer/collective/consumer dependencies. Submission advances only after the
communication backend confirms its work has been enqueued; this is **not**
completion and does not release any operand.

The private native owner preallocates bounded pending-lease storage. Enqueued
collectives retain their buffers, stream and communicator until the terminal
stream fence retires the entire frame. Pending work rejects close/rebinding;
cancellation/failure stops submission and follows explicit abort/drain cleanup.
No steady-state allocation or numerical-policy change is needed.

Graph replay captures compute intervals only, with eager collectives between
them. This avoids treating NCCL capture support or dynamic row counts as an
implicit capability. All graphs and argument leases are startup-owned; fixed
launch geometry consumes mutable request metadata through existing buffers.
CUDA handles, capture ABI and events stay in the device backend. A bounded
graph reservation and actual selected-mode reporting are required. No fallback
may silently claim replay performance.

## Results

### Same binary, separate changes

All native modes below use ARM executable `aed93d05` and the same retained
FP4-staging AOT. One warmup is excluded, C1 sequential requests; prompt lengths
`[22,23,22,23,24,36]`, generated lengths `[4,4,8,8,8,3]` (35 total).

| Mode | Six-request completion sum | Relative to vLLM |
| --- | ---: | ---: |
| Per-collective whole-stream retirement | 9.855 s | 3.115x |
| Queued TP, eager compute | 8.938 s | 2.825x |
| Queued TP, graph compute (two cohorts) | 8.731 / 8.739 s | 2.760 / 2.762x |
| Retained vLLM | 3.163588 s | 1.000x |

The first change reduces time **9.305%**; graph replay adds **2.316%** on the
queued mode, for **11.405%** total; the repeat gives 2.226% additional and
11.324% total. These are sequential cohort observations,
not a confidence interval or a claim to have closed the remaining gap.

| Prompt | Input/output tokens | Per-site (s) | Queued (s) | Graph (s) | vLLM (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Fact | 22/4 | 1.200 | 1.078 | 1.063 | .385331 |
| Arithmetic | 23/4 | 1.219 | 1.099 | 1.078 | .396595 |
| Chinese | 22/8 | 2.067 | 1.890 | 1.840 | .656153 |
| JSON | 23/8 | 2.090 | 1.919 | 1.870 | .670817 |
| Code | 24/8 | 2.149 | 1.928 | 1.869 | .673605 |
| Retrieval | 36/3 | 1.130 | 1.024 | 1.011 | .381087 |

Native CLI completion includes diagnostic output/retirement; reference timing
is buffered HTTP completion including client/server overhead. Native uses
BF16 KV, vLLM uses FP8 MLA KV. Four reference texts agree after removing the
native stop marker; Chinese/JSON differ despite matched output lengths.
Reference token IDs are unavailable. All native output IDs agree across modes.
This compares implementations on literal matched prompts, not bitwise numerical
equivalence, full-model correctness, or long-context/batched throughput.

Literal baseline root: `/tmp/lunaflux-model-bench-glm-tp-each-20261013-v1`.
Reference root: `/tmp/lunaflux-reference-bench-vllm-glm-staging-20261014-v1`.
The reference is retained, not rerun in this change; its suffix is an experiment
identifier, not evidence of a future measurement date. All six HTTP responses
succeeded; the subsequent deliberate container teardown had rank 1 exit 137,
so no clean reference-shutdown claim is made.

### Earlier cohorts and validation

Retained baseline:
9.756/10.189 s summed completion across six sequential requests; vLLM 3.163588 s.
Native uses BF16 KV; the reference uses FP8 MLA KV and breakable CUDA graphs.
These are not identical numerical implementations or a correctness oracle.

Queued submission implementation `712b6fec` passes 189 affected tests and the
native ASan/UBSan lifecycle gate, including delayed submission, pending GPU
completion, retained operands and async failure. Its first actual six-prompt
run at `/tmp/lunaflux-model-bench-glm-tp-queued-20261013-v3` completes in
8.897 s: 8.8% below the retained 9.756 s cohort, still 2.8123x the vLLM cohort.
All 35 emitted IDs match the old native run. Both ranks exit 0, task swap is
zero and minimum host available memory is 23.0006 GiB. The repeat at
`/tmp/lunaflux-model-bench-glm-tp-queued-repeat-20261013-v1` takes 9.631 s.
Its first request is 1.825 s versus 1.070 s in the first cohort; the other five
total 7.806 s versus 7.827 s. Preserve this outlier: two cohorts do not establish
a confidence interval, and the first 8.8% delta alone is not a stable estimate.
Both cohorts use the first queued binary; same-binary comparisons follow below.
Earlier v1/v2 preparation attempts are preserved: missing staged binary, then
an AOT symlink rejected by the existing file loader before GPU execution.

The graph-stage allocation regression caught tuple-return and iterator boxing
in the new host loop. Scalar IR accessors and indexed traversal remove it;
122 packed execution tests pass, including zero-allocation eager/queued/graph
loops and explicit in-flight abort. This also fixes the tuple boxing in the
first queued benchmark binary; subsequent modes must use the same revised
binary to isolate submission from graph effects. Graph capture uses the existing device
executor, one captured compute interval per collective cut, with no new CUDA
kernel or NCCL capture dependency. Its existing host ABI ASan/UBSan
lease/race/retry gate passes. Source-pinned ARM build `aed93d05` succeeds with
226 tasks and 40.334 CPU seconds. Full native check and **5,145/5,145 tests**
pass with the existing warning exclusions `-20-25-29-35-79-92` unchanged.

First graph cohort `/tmp/lunaflux-model-bench-glm-tp-graphs-20261013-v1`:
both ranks report 133 prepared graphs, exit 0 and close their CUDA contexts.
Completion times are `[1.063,1.078,1.840,1.870,1.869,1.011]` s, total
8.731 s; TTFT `[.454,.473,.448,.488,.484,.603]` s. All output IDs match the
retained native run. Peaks are 57,656,545,280 / 67,932,684,288 bytes, task
swap 0, minimum host available 22.9432 GiB. This first run is 2.7598x the
retained vLLM cohort; it is not a parity claim. Same-binary queued-only and
short scoped traces follow.

The first same-binary queued restart (`...tp-queued-scalar-20261013-v1`)
fails before GPU loading with `Address already in use` shortly after the
previous listener closes, 9,207,808-byte peak and zero swap. Its journal is
retained; no GPU result exists. Restart v2 uses a new output root after the
socket retires.

Same-binary queued cohort `...tp-queued-scalar-20261013-v2` takes 8.938 s,
with TTFT sum 2.986 s and post-first-token sum 5.952 s. The graph cohort's
corresponding sums are 2.950 / 5.781 s. Thus the first isolated graph comparison
is **2.316% less completion time**, not the entire ~10% baseline improvement.

## Dispatch trace (separate from timing cohorts)

Short traces use one 16-input/two-output prompt plus one excluded-from-timing
warmup: four model steps. Both trace modes explicitly include all CUDA APIs
(`cuStreamQuery` is absent from the default Nsight API set) and graph nodes.
The profiler substantially perturbs execution; its durations are **not**
unprofiled latency estimates. No profiler or comparison enters the compiler.

Queued trace: `/tmp/lunaflux-model-bench-glm-tp-queued-trace-20261013-v1`.
Graph trace: `/tmp/lunaflux-model-bench-glm-tp-graphs-trace-20261013-v2`.
Analysis roots are `/tmp/lunaflux-glm-submission-trace-queued-20261013-v2`
and `/tmp/lunaflux-glm-submission-trace-graphs-20261013-v1`.

Both ranks execute all 132 collectives per step with **zero StreamQuery calls
between the first launch and the last collective enqueue**, on all four steps.
Graph mode executes **133 cuGraphLaunch calls per step** on both ranks. The
2,020/2,022 compute kernels per rank per step and 132 collectives are unchanged;
graph replay reduces host dispatch, not mathematical work. StreamQuery still
runs at final completion and retains resources until then; "one frame fence"
does not mean just one polling API call.

The queued ingress trace's final decode has 5.972 ms of kernel-free intervals
before compute kernels versus 0.966 ms with graphs. Intervals before collectives
remain 37.694 / 28.590 ms respectively. Egress differs, and NCCL kernel time
includes peer waiting: these quantities cannot be summed across ranks or used
as additive unprofiled speedup claims. Host enqueue spans remain substantial
even without StreamQuery; their remaining cause is not established by a CUDA
API trace alone. Native NCCL uses nonblocking submission, while the inspected
local vLLM PyNccl initializes a conventional communicator. This is a further
measurement hypothesis, **not a demonstrated explanation for the whole gap**.

The first graph-trace startup also encountered the retired socket's TIME_WAIT
before GPU loading. Its failure root is retained. The external benchmark
driver now waits for its two fixed ports to retire without changing runtime
socket semantics. The initial analysis returned no ingress steps because that
rank has no vocabulary head; the corrected external query uses the next
embedding/final kernel as that rank's endpoint. Its original empty report is
retained, not interpreted as missing GPU work.

### Matched old-fence trace confirms the cause

The same `aed93d05` binary with per-collective retirement produces
`/tmp/lunaflux-model-bench-glm-tp-each-trace-20261013-v1`; analysis is
`/tmp/lunaflux-glm-submission-trace-each-20261013-v1`. Its final decode step:

| Observed quantity | Per-site wait | Queued TP | Queued + graphs |
| --- | ---: | ---: | ---: |
| Ingress stream queries before final collective enqueue | 61,444 | 0 | 0 |
| Egress stream queries before final collective enqueue | 255,594 | 0 | 0 |
| Collectives per rank | 132 | 132 | 132 |
| Actual graph launches per rank | 0 | 0 | 133 |
| Egress compute-kernel sum (profiled ms) | 181.816 | 183.961 | 182.985 |
| Egress host enqueue span (profiled ms) | 505.457 | 258.775 | 243.612 |

Subsequent full-step accounting found that this analyzer ended egress at the
vocabulary head and omitted the following greedy kernel. The graph trace's
complete final-decode compute sum is **193.504 ms**, including **10.518 ms**
greedy, rather than 182.985 ms. The table above retains the original endpoint
for historical mode comparison; it is not the complete token-step compute
cost. Unprofiled completion results are unchanged. See
[the remaining-gap diagnosis](GLM_REMAINING_GAP_2026-10-14.md).

The invariant compute work and eliminated pre-final stream queries establish
the targeted dispatch change. The old trace records 1,524,636 egress stream
queries over four steps (475.272 ms *inside the API calls alone*); tracing those
polls itself greatly perturbs the schedule. Consequently the ~halved profiled
host span must **not** be advertised as a ~2x end-to-end gain. The unprofiled
cohorts measure ~9% for queued submission and a small additional graph gain.
NCCL duration also changes with peer arrival and tracing; it is not an isolated
network-bandwidth measurement.

These additions are explicit policies, selected end-to-end by the benchmark's
`--queued-tp` and `--graph-tp` modes. Existing eager/default capability coverage
remains available; no unsupported backend is silently claimed to capture.
The changed native lifetime boundary passes host ASan/UBSan and deterministic
lease/cancellation tests. These are not a claim of a new full-model CUDA
Compute Sanitizer campaign; the existing NCCL/CUDA sanitizer-version limitation
is unchanged. All completed physical modes explicitly close both contexts.

## Final repeat and handoff

`/tmp/lunaflux-model-bench-glm-tp-graphs-repeat-20261013-v1` completes with
per-prompt times `[1.054,1.076,1.858,1.871,1.875,1.005]` s, total 8.739 s.
TTFT sum 2.942 s; post-first-token sum 5.797 s. All 35 native output IDs remain
unchanged and both ranks report 133 prepared graphs. Mean of the two graph
cohorts is 8.735 s (about 4.01 output tokens/s including prefill, not decode-only).
The retained reference is about 11.06 output tokens/s on the same length vector.

Repeat memory peaks are 67,285,569,536 / 69,569,478,656 bytes, task swap 0,
minimum host available 22.9051 GiB. All successful task memory peaks in these
new cohorts/traces are below 66 GiB; allocation plans remain below 100 GiB with
the graph reservation inside the existing 2 GiB runtime reserve. Both 100 GiB
process ceilings, zero-swap limits and 16 GiB host reserve guard remained active.
Final GPU-owner checks on both Sparks are empty. No payload hashes were computed.

Implementation is committed in `712b6fec` and `aed93d05`; diagnostic dispatch
measurement is `a21f0e2e`. Source-pinned ARM builds and the local 5,145-test suite
pass, as do the changed NCCL and existing ordered-graph host ASan/UBSan gates.
The comparison/trace tools remain external `.mbtx` automation. Unrelated dirty
toolchain/TLS work is preserved. The measured remaining ~2.76x completion-time
ratio is not solved or attributed wholesale to NCCL, kernels, or the compiler.
