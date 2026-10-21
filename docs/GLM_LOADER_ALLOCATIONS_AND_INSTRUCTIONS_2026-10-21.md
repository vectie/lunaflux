# GLM loading allocations, projection instructions and two-rank retrace

## Outcome and scope

The loading trace identifies a real staging allocation and the replacement
reader removes it. **Full vLLM loading still fails the early reserve guard;
there is no fresh vLLM decoder trace.** Two projection alternatives substantially
reduce instructions, but neither passes the paired latency gate across the
tested row counts. They are archived, not selected in serving.

The accepted LunaFlux path was freshly retraced on both ranks, cold and warm,
with all 45 layers and 90 TP reductions in the selected 64-token chunk. The
reference decoder comparison below is explicitly the retained October 19 trace.
This is not an overall throughput benchmark. No payload hashes, relaxed memory
limits, changed references, arithmetic tolerances or unqualified deployment.

The physical campaign was bounded to two loading trials and two projection
candidates. Unsuccessful experiments are retained rather than reported as fixes.

## Loading: from suspected lifetime to an observed allocation

The installed local Runai iterator clones each borrowed reader tensor before
passing it to vLLM. Deleting that clone alone is unsafe: model consumers can
retain weight/scale pairs after advancing the iterator. The previous weak-reference
probe did not explain growing anonymous memory. This round adds `mallinfo2`,
the actual reader-buffer address/extent, and periodic process `smaps` snapshots.
These are external loading diagnostics, not compiler or production runtime code.

Observed source ownership is
`SafetensorsStreamer → DistributedStreamer → FileStreamer → requests_iterator.buffer`.
With local mode, concurrency 1 and `memory_limit=0`, that buffer is sized for the
largest tensor across the file list, not for the current tensor. Here its extent
is **1,268,776,960 bytes (1.182 GiB)**, even while ordinary expert tensors are
only a few MiB. Rank 0's logged address lies inside the anonymous mapping
`ee0820600000-ee086c000000`; its size is 1,239,040 KiB and observed RSS/PSS is
1,239,024 KiB. The mapping becomes fully resident while loading progresses.

At index 0→15,360, worker anonymous PSS rises 1,618,092→2,923,212 KiB. In the
same interval, libc arena size changes only 757,702,656→759,865,344 bytes and
live arena bytes 573,402,272→573,903,472. This directly locates the large reader
mapping; it is not evidence of a 1.3 GiB growing malloc freelist, tensor-object
leak, or CUPTI allocation. Shared-memory PSS is not assumed to be POSIX `/dev/shm`
and is not added to device accounting as independent physical memory.

Evidence:

- `/tmp/lunaflux-reference-bench-glm-loader-maps-20261021-v1`
- `loader-owner-431900.jsonl`, `loader-owner-2950818.jsonl`
- `loader-maps-431900-15360.txt`
- Installed original and instrumented source: `/tmp/lunaflux-loader-observer-maps-20261021-v1`

### External replacement and measured effect

`scripts/prepare-glm-owned-reader.mbtx` accepts the installed source explicitly
and replaces only the local Runai iterator body. It reads safetensors headers,
then reads each payload directly into independently owned CPU storage. Consumer
tensor views can outlive the iterator without reusing another tensor's buffer.
It neither allocates a checkpoint-wide maximum reader buffer nor clones a
borrowed payload. Checkpoint bytes, tensor shape/dtype and weight consumers stay
unchanged. Distributed streaming is not claimed by this local diagnostic.

The CPU-only fixture retains every returned tensor until after iterator
retirement. BF16/F32/U8/I64/E4M3 payloads from two files are byte-identical, with
matching shapes and dtypes. This is a bounded ownership fixture, not proof that
the incomplete full-model load is correct. `scripts/check-glm-owned-reader.mbtx`
retains the fixture without a GPU dependency.

At **identical observed tensor names and indices**, index 15,360:

| Worker observation | Original reader | Owned reader |
| --- | ---: | ---: |
| Rank 0 anonymous PSS, KiB | 2,923,212 | 1,428,332 |
| Rank 1 anonymous PSS, KiB | 2,922,016 | 1,433,044 |
| GPU allocated, bytes, each rank | 94,996,518,912 | 94,996,518,912 |

The owned reader also bypasses Runai's initialization, so the entire PSS delta
is not attributed exclusively to the buffer. The buffer's own mapping/extent
is separately observed. The last rank-0 sample reaches index 130,176 with
1,533,728 KiB anonymous PSS. Large individual tensors still have a transient
cost: `lm_head.weight` and `embed_tokens.weight` are each 1,268,776,960 bytes;
rank 0's post-yield anonymous PSS is 2,920,472/2,672,504 KiB at those frontiers.
The replacement removes permanent maximum-size staging, not all large tensors.

| Loading trial | Host minimum R0/R1, GiB | GPU maximum R0/R1, GiB | Result |
| --- | ---: | ---: | --- |
| Mapping/allocator observer | 17.566 / 19.133 | 89.628 / 89.628 | Early guard; retired |
| Independent owned reader | 17.305 / 20.047 | 89.628 / 89.628 | Early guard; retired |

The existing 18 GiB early guard remained unchanged ahead of the user's 16 GiB
floor. Samples remain above 16 GiB; these are sampled envelopes, not guaranteed
continuous minima. Container memory and memory+swap remain 96 GiB, with no
additional swap. Exit 137 follows requested retirement and is not an OOM claim.
The owned reader made much more loading progress, but **did not reach readiness**.

Owned evidence: `/tmp/lunaflux-reference-bench-glm-loader-owned-20261021-v1`,
including `ownership-summary.json`, `window-reserve.txt`, `memory-envelope.json`,
`WINDOW_RESULT.txt`, both worker logs and terminal unit journals. The final
rank-0 host-availability fall is not fully attributed: the last sampled worker
PSS is flat, while host anonymous/slab accounting changes. No surviving snapshot
identifies every allocation owner in that interval. Do not relabel it as another
reader leak, cache issue or GPU-memory failure without that observation.

## Projection: what the 4.46× number actually refers to

The previously quoted **4.46×** instruction ratio belonged to the *rejected*
M64/N128/K32 traffic-reduction candidate, not the accepted serving kernel.
The accepted M32/N64/K64 baseline executes 17,119,232 warp instructions, versus
the retained matched reference leaf's 2,293,848 (7.46×). Keeping those baselines
separate is necessary to avoid inventing a production improvement.

Local source `../vlm/vllm/model_executor/layers/linear.py` routes
`UnquantizedLinearMethod.apply` through `dispatch_unquantized_gemm`; on this CUDA
path `layers/utils.py::default_unquantized_gemm` uses `torch.nn.functional.linear`.
The actual installed selected leaf was measured as
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`.
The closed nvjet implementation is not readable source in `../vlm`; its retained
SASS, not a guessed library name, establishes uniform TMA coordinates, swizzled
loads, fragment reuse and phase waits. Grid 256 alone does not prove a split-K
factor or its useful output decomposition.

The accepted generated SASS recomputes row/lane validity, swizzled destinations
and 64-bit source pointers inside each K-epoch copy loop. The new experiment
represents invariant operand ownership in immutable `MatrixFragmentPipeline`;
private CUDA lowering computes addresses once, advances only K/slot offsets,
and unrolls fixed copy atoms. The matrix fold remains ascending K16 F32 with
BF16 publication and two shared/register slots. There is no new compiler
reference comparison or model-specific hardware branch.

Candidate 1 keeps M32/N64/K64. Candidate 2 also widens M to 64. Actual exported
source, compiled fixture and counters are retained in:

- `/tmp/lunaflux-glm-projection-staging-invariant-export-20261021-v1`
- `/tmp/lunaflux-glm-projection-staging-invariant-20261021-v1`
- Accepted SASS: `/tmp/lunaflux-projection-baseline-sass-20261021-v1.txt`

### Same-campaign counters: M64/N4096/K8192, seed 713

| Metric | Accepted | Invariant M32 | Invariant M64 |
| --- | ---: | ---: | ---: |
| Warp instructions | 17,119,232 | 5,773,824 | 5,042,688 |
| L2 read sectors | 6,294,244 | 6,294,268 | 4,197,456 |
| Registers/thread | 46 | 62 | 98 |
| Shared allocation, KiB | 24 | 24 | 32 |
| Register-limited CTAs/SM | 10 | 8 | 4 |
| Shared-limited CTAs/SM | 4 | 4 | 3 |
| Active warps, % peak | 22.16 | 22.39 | 11.43 |
| Tensor activity, % peak | 10.07 | 9.77 | 10.32 |
| Long-scoreboard share, % | 64.52 | 69.74 | 64.71 |
| Barrier share, % | 22.73 | 22.62 | 21.34 |
| Short-scoreboard share, % | 1.13 | 1.34 | 1.80 |
| Profile duration, us | 377.760 | 373.280 | 372.256 |

Candidate 1 removes 66.3% of instructions with unchanged traffic and similar
active-warps measurement. **The kernel does not become proportionately faster.**
This intervention rules out the extra address instructions as a sufficient
explanation for its latency. It does not prove that instructions have zero cost
or that all remaining time is DRAM bandwidth. Stall percentages are active-cycle
shares, not additive wall-time allocations. Candidate 2 reduces reads as well,
but raises resource demand and does not qualify either.

### Paired unprofiled replay, separate from counter collection

Five alternating pairs, 20 graph replays per pair. Active/inactive output bytes
match and the existing sampled F64 oracle passes for rows 0/1/3/22/36/64/65.
These rejected candidates were not put through sanitizer/serving qualification.

| Live rows | Accepted→M32 median, us | Accepted→M64 median, us |
| ---: | ---: | ---: |
| 1 | 290.882→288.293; inconclusive | 295.829→295.173; inconclusive |
| 3 | 310.083→310.125; regression | 301.147→302.206; inconclusive |
| 22 | 304.813→310.259; regression | 305.474→310.694; regression |
| 36 | 308.466→313.515; regression | 304.509→307.069; regression |
| 64 | 323.414→309.275; inconclusive | 313.918→324.242; regression |

For M32/64 rows, the ratio of medians improves 4.37%, median paired gain is
3.65%, but minimum paired gain is only 0.51%, below the existing 1% gate.
Other row counts regress. Row 1 uses the unchanged decode branch and does not
establish a decode improvement. No gate or tolerance was relaxed.

Both candidates are removed from executable packages and preserved as
`benchmarks/experiments/glm_invariant_addresses_20261021.patch` against `48282e7c`.
It passes `git apply --check`. No new projection artifact was deployed.

## Fresh two-rank execution trace

Root: `/tmp/lunaflux-glm-http-owned-followup-20261021-v1`.
Accepted runtime `ed6be466`, AOT `b1788d1d`, 16-site collective graph windows,
64-token physical chunk, eight slots, context 8192, admission 16. The two
requests each report 7,973 prompt tokens, zero cached tokens and one output
token. A separate complete arithmetic request answers `323` with stop finish.
All trace loss/error/refused counters are zero (rank records 13,778/22,078).

Both selected passes contain the complete 45-layer decoder and 90 collectives.
The boundary is first HC pre through final HC post: embedding, vocabulary head,
sampling and HTTP are excluded. The shared expert runs alongside routed work;
they join before local combine and one MoE reduction. First three FFNs are dense.

| Rank/pass | Decoder, ms | TP union, ms | Shared overlap, ms | Shared uncovered, ms |
| --- | ---: | ---: | ---: | ---: |
| R0 cold | 277.537 | 20.241 | 108.988 | 0.139 |
| R1 cold | 277.141 | 18.327 | 108.324 | 0.162 |
| R0 warm | 276.170 | 18.987 | 108.855 | 0.091 |
| R1 warm | 275.810 | 17.096 | 109.344 | 0.097 |

No compute/TP overlap is observed in these decoder intervals. Rank-1 cold
disjoint wall allocation below counts overlapping shared/routed work only once.

| Stage | Fresh LunaFlux R1, ms | Retained vLLM R1, ms | Difference, ms |
| --- | ---: | ---: | ---: |
| KDA projections | 34.441 | 26.705 | +7.736 |
| Sparse projections | 15.837 | 9.373 | +6.465 |
| HC | 6.787 | 3.950 | +2.837 |
| KDA other | 8.815 | 4.423 | +4.393 |
| Dense FFN | 5.824 | 2.282 | +3.542 |
| TP including waits | 18.327 | 17.469 | +0.858 |
| MoE union | 183.183 | 184.459 | −1.276 |
| Sparse core/index | 1.851 | 2.116 | −0.265 |
| Kernel-free / cross-family overlap | 2.075 | 5.294 | −3.219 |
| **Decoder** | **277.141** | **256.071** | **+21.070** |

Projection allocation is 50.279 versus 36.078 ms, a 14.201 ms difference.
That is measured allocation against a historical trace, not a guaranteed gain
or a controlled regression from the prior 274.487 ms native capture. This
round deployed no change that could support a production speedup claim.

### TP and chunk boundaries

Paired collective alignment uncertainty is 0.258 ms, so rank arrival differences
are intervals, not synchronized-clock point estimates. Rank 0's first cold
collective takes 2.246 ms; 1.448–1.706 ms is bounded as early arrival. Rank 1's
corresponding interval is 0.656 ms. Warm first intervals are 0.583/0.201 ms.
These are not inferred wire-transfer times.

The first two dense-FFN reductions on rank 1 are 0.193/0.174 ms cold and
0.200/0.174 ms warm. The old 3 ms stall does not reappear there. Across all sites,
the largest interval after both ranks start is bounded below 0.798 ms cold and
0.333 ms warm. Broad summed early-arrival bounds are retained in the JSON; they
must not be treated as a precise additive network budget.

Rank-1 decoder→next-chunk handoff is 4.194 ms cold / 3.954 ms warm; 3.860/3.618 ms
has no recorded device work. Rank 0 is 3.780/3.806 ms, with 3.447/3.476 ms
uncovered. This filtered trace alone does not identify the CPU-side cause.

Historical reference root remains
`/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2/rank1.sqlite`.
No fresh rank-0 vLLM pass exists. Both fresh loading failures are explicitly
retained, not replaced by the older trace without a label.

Trace sampled memory maxima: GPU 94.241/94.270 GiB, host minima 17.092/18.911 GiB,
ranks 0/1; owned swap zero. Failed reference owners and native trace owners were
retired before restarting the accepted GLM service.

## Validation and remaining work

Owned-reader source tests 2/2, observer 1/1, CPU fixture, and standalone fixture
check pass. After withdrawing the kernel changes, grouped-matrix/tiled-source
tests pass 37/37. Native check passes with the existing migration-warning
exclusions `-20-25-29-35-79-92`; this is not a new clean full-release-suite claim.
Scoped `moon info` reports 121 existing migration warnings and zero errors.
Only the three task scripts were formatted; unrelated migration files remain.

Still unresolved: the last loading reserve loss outside the observed reader
mapping, fresh vLLM two-rank capture, and a projection implementation that
reduces latency rather than only counters. Next measurement needs simultaneous
allocation ownership during that last host-memory fall and instruction-local
wait attribution for the dependency/retirement schedule. Neither repeating the
same geometry nor blindly removing barriers is justified by this evidence.

Restoration is complete. Evidence is recorded in
`/tmp/lunaflux-glm-loader-maps-restoration-20261021-v1`: both accepted units are
active, private health is healthy, and the complete arithmetic response is
`323` with `finish_reason=stop`. The sampled restoration GPU maxima are
94.239/94.268 GiB and host minima 17.074/18.947 GiB, ranks 0/1; owned swap is zero.
The observer retired after saving the terminal units, health and answer.
This verifies service restoration, not a public-key test or broad model quality.
