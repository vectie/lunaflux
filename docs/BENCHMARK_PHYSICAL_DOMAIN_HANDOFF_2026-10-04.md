# Physical domain pruning and worker handoff measurements

The three requested experiments were implemented and measured: inactive
fused-QKV fragment-row pruning through the shared physical IR, bounded worker
completion/publication coalescing, and fresh mixed-attention and matched
projection-chain traces. The combined changes produced **no measurable
end-to-end improvement**. An isolated partial-tile QKV improvement did not
translate into a comparable serving improvement.

These measurements use Qwen3-0.6B BF16 on the Spark GB10, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, CUDA 13.0.88. GPU work was serialized.
Qualification/calibration had an 8 GiB memory ceiling; serving and reference
profiling had a 64 GiB ceiling, no swap and a 32 GiB MemAvailable reserve.
The ordinary paired campaign's minimum MemAvailable was approximately 98.76 GiB.

## Implemented changes

Commit `654e3269` introduces the opaque immutable `FragmentRowDomain` in
`compiler/physical_tile_ir/fragment_rows.mbt`. The shared projection pipeline
carries this domain. CUDA ingress lowering consumes the same active prefix for
operand staging, fragment loading, MMA and publication. Partial fragment rows
are still zero-filled; inactive fragment groups do not perform projection work.
The fixed numerical fold, scalar single-token route and epilogue semantics remain.

The complete projection lifetime is specialized once, outside the reduction
loop, rather than selecting live row counts on each reduction iteration. An
earlier per-iteration version was rejected after a full-tile regression and
retained separately on the test host. Selection remains a pure physical-domain
transformation; the CUDA constant specialization is its device lowering.

Commit `9e545039` coalesces bounded progress transitions:

- The root-bound exchange can finish writing and consume completion/telemetry
  without an extra empty `Pending` transition. It performs at most one syscall
  per phase and returns immediately when IO would block.
- The reactor-only lease path can start, observe completion and commit that
  exact retained flight in one owner turn. The public single-transition API
  retains its cancellation cuts.
- Session publication can follow an advanced worker transition in the same turn.
  Caller acknowledgement, output credit, cancellation, KV commitment and the
  single-flight boundary remain authoritative. There is no speculative next
  submission, new global state or diagnostic work in the production token path.

The Linux real-process regression covers cancellation before completion and
after publication, submission rejection across an unconsumed publication,
ordinary two-token positions 0 and 1 after consumption, terminal handling and
clean shutdown. Its echo fixture now consumes the existing absent-attestation
bootstrap frame before reporting Ready. No new production wire format was added.

## Artifact propagation and correctness

The tested source is the previously qualified serving source plus a clean
overlay of these owned changes, not an assertion that every unrelated change in
the dirty checkout was included in a commit-pinned release.

| Artifact | SHA256 |
| --- | --- |
| Prior selected ingress CUBIN | `418192ff86c014dc38af3d4dce8596bfab84ab55b7ba308e9795658e6cc25df4` |
| New selected ingress CUBIN | `e1256b6e7e3fdeefed76eb76513e1c3b0899665fbdb59fd7671f398eb2aaca07` |
| New ordinary serving worker | `d3b4d60d1eacb6698209fc14f88061b3a47c1fa9ac0db3c306eee454f9cc77b2` |
| Diagnostic row-marker worker | `da14f8c7f74ef47a5b5704efa00533cd8286e75947efd2ea5267cd019074fab2` |

Deployment module 7 contains the new ingress CUBIN, and its actual kernel root
points to the new artifact tree. The trace uses a separate row-marker worker;
its AOT kernels are unchanged. The selected tile schedule digest remains
`384524c01d71b50b647db543ebe98bcc913e89b50aa2d548c566f6be87659b4e`.

The ingress vectors `[1,2,8,16,17,32,33,48,49,64,65,2048]` passed two independent
five-sample comparisons with bitwise-equivalent outputs. Tail-65 memcheck with
full leak checks, racecheck and synccheck passed. Racecheck uses one analysis
worker to remain inside the memory ceiling; this does not reduce GPU test work.

Local native tests passed 4,401/4,401. The current migrated checkout still needs
warning exclusions `-79-20-29-25-92-14`; this is not a claim of a globally
warning-clean migration. `moon info` and formatting checks also passed.
New automation has focused archive-selection, median and chain-label tests.
The generated native coalescing helper uses stack result structures; this
inspection is not a replacement for a full allocation campaign. Native macOS
socket lifecycle execution remains unsupported, so Linux supplies the process
regression rather than a claimed macOS process pass.

Changing the ingress CUBIN invalidated the existing aggregate attention-route
scope. Fresh attention probes and unchanged-attention sanitizers were run and
the routes were rebound to the actual new export report. Admission was not
weakened and old latency records were not relabeled as new measurements. Failed
calibration, export-metadata and launcher attempts remain on the host.

## Ordinary fresh start paired serving timings

Three fresh trials per arm alternated old/new, new/old, old/new. Each cell had
warmup and identical input token vectors. Throughput counts generated output
tokens across all 16 requests, not input plus output tokens.

| Input output concurrency | Prior median ms | New median ms | Prior output tok/s | New output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 128 32 C16 | 355 | 355 | 1442.25 | 1442.25 |
| 4096 64 C16 | 4538 | 4543 | 225.65 | 225.40 |
| 4096 256 C16 | 12642 | 12662 | 324.00 | 323.49 |

The differences are zero, +0.11% and +0.16% completion time: no demonstrated
gain, and no demonstrated material regression given the observed trial spread.
Both long-input cells have agreeing output token vectors across both arms and
all three trials. Short-input outputs vary in the prior arm as well as the new
arm, including request rows 0 and 12. The aggregate report therefore correctly
records `token_agreement=false`; the short cell is not a deterministic-output
qualification, and this comparison does not establish its cause.

These are ordinary serving timings. Profiler replay durations below must not
be substituted for them. Fresh reference traces were collected this round,
but fresh **ordinary** vLLM/SGLang timings were not collected by this experiment.

## Isolated QKV versus serving QKV

The isolated 16-token median falls from approximately 30.75 to 20.52 microseconds,
a 33.3% improvement. Its replay counters change as follows:

| Metric | Prior | New |
| --- | ---: | ---: |
| Replay duration microseconds | 94.976 | 59.936 |
| Warp instructions | 1,311,808 | 915,392 |
| Shared wavefronts | 1,298,176 | 689,664 |
| Excessive shared wavefronts | 920,320 | 440,576 |
| Registers per thread | 116 | 116 |
| Local spilling requests | 0 | 0 |

Full 2048-token replay instructions are effectively unchanged:
52,047,872 versus 52,101,120. Its excessive shared wavefront count remains
24,903,680. The change removes inactive row work, not full-tile work or all
shared-memory conflicts.

In the serving traces, the same small-grid QKV symbol executes 7,140 times:
414.50 ms in the prior trace versus 409.10 ms in the new trace. The full-grid
calls total 387.37 versus 385.29 ms, with the same 896 calls. This is only about
7.5 ms less QKV activity across a roughly 12.65-second trace, not a 33% serving
gain. These trace totals are diagnostic observations, not randomized estimates
of isolated causal effects.

The source and row markers confirm that step counts carry actual token counts,
not the graph bucket's upper bound. The new artifact did propagate. However,
the isolated fixture and live serving differ in operands, row/page distribution
and surrounding kernels. Their cache/dependency contribution has **not** been
isolated; no specific cache bottleneck is proven by these results.

## Completion publication and submission boundary

Signed CUDA API timestamps bound to graph GPU envelopes give:

| Diagnostic handoff component | Prior ms | New ms |
| --- | ---: | ---: |
| Inter-step GPU gaps | 380.875 | 381.177 |
| GPU completion to event return | 2.460 | 2.491 |
| Device-to-host read span | 2.831 | 2.840 |
| Read completion to next upload | 311.469 | 312.187 |
| Host-to-device upload span | 16.906 | 17.110 |
| Graph launch API | 44.729 | 43.943 |

The coalescing removes state-machine yields, but it has not measurably reduced
the serving boundary. The 312 ms interval includes child/parent transport,
scheduling, output rendering and unstamped CPU work; it cannot be called pure
IPC or scheduler time. Negative launch tails are retained where launch API
execution overlaps GPU start.

## Exact mixed attention and projection chains

Each work signature is the sorted vector of `query_tokens:past_tokens`, not
just a concurrency label. The fresh LunaFlux and vLLM traces have equal totals
of 69,616 query tokens and 151,484,416 causal query-key pairs per query head.
They share ten exact work vectors. SGLang packs work differently and has no
exact-vector intersection in this capture; no matched-invocation SGLang result
is inferred from an equal-looking grid.

For the exact mixed vector `1:4097,2047:2047`, the following totals cover all
28 layers of one step. vLLM projection roles are derived from pinned model
source order, attention/SiLU anchors and complete projection counts, rather
than assigning output/down from a generic matrix symbol's grid.

| Whole chain | LunaFlux ms | vLLM ms | Difference ms |
| --- | ---: | ---: | ---: |
| Attention including split decode and merge | 27.876 | 23.315 | +4.561 |
| QKV including normalization RoPE and KV write | 12.669 | 11.898 | +0.770 |
| Output projection | 5.257 | 3.265 | +1.992 |
| Gate/up including activation | 15.850 | 13.045 | +2.805 |
| Down projection | 6.798 | 3.983 | +2.815 |

Our attention chain has 28 prefill calls plus 28 decode partial and 28 merge
calls; vLLM has 28 combined attention calls. Our prefill portion is 24.954 ms
and the separate decode/merge portion is 2.922 ms. Thus the matched attention
excess is not all inside the prefill kernel. The source-bound projection
analysis resolves 286 reference steps; two ambiguous steps are excluded rather
than force-labeled.

The complete new trace spends 9,130 ms in attention, 2,543 ms in decoder
projection/postops and 401 ms with no observed GPU activity. These are exclusive
activity categories, not a sum of overlapping kernel durations. The requested
row pruning and progress coalescing leave most of that work unchanged.

Live Nsight Compute attempts in both a bounded container and a bounded
administrator host unit aborted during startup, before logical work markers.
The replay result is therefore **not available** for this mixed serving
invocation. The empty child diagnostics do not identify an underlying admission
or profiler-injection cause. Existing isolated counters above are not presented
as new live mixed-attention counters. Failed attempts and their journals remain
preserved; diagnostic-only launcher tools were removed from the local tree.

## Reproduction artifacts

The completed remote roots under `/home/wlc004s/` are:

- Qualification: `lunaflux-domain-handoff-v2-20261004.y02hFOt0`.
- Serving artifacts and fresh routes: `lunaflux-domain-serving-v4-20261004.QmE9WZ7s`.
- Alternating timings: `lunaflux-domain-paired-v5-20261004.IqaFfEhn`.
- Fresh three-engine traces: `lunaflux-domain-handoff-trace-20261004.hlmKsGE4`.

Non-overwriting measurement archives were downloaded to
`/tmp/lunaflux-domain-results-20261004.Y0NDGtEU/{qualification,serving,paired,trace}`.
All archive hashes and every extracted `MEASUREMENT.sha256` entry verified
locally. Model weights and build caches are excluded; relevant CUBINs, reports,
recipes, worker binaries, ordinary request records and profiler traces are kept.

| Archive | SHA256 |
| --- | --- |
| Qualification | `85000321a61cdb946a21673b1d31ee60d4aad8e4945d7b937b2d3f55f2d85f6b` |
| Serving | `31d07b93a26846f70cb512fc3fcb4a2d6c00221a8ceab21ae60516d8abdca63e` |
| Paired timings | `7d99f3bb7a10b6af024bbc61efaf408cd9d1b8dec76ca7c8ff76f2273c3ffbbb` |
| Traces | `48915f97b2bb08ecefa7c42bf4c182c93dac44cf5177f02efae21ec80131793a` |

Failed live counter roots are `lunaflux-domain-exact-mixed-20261004.iYvGINz7`
and `lunaflux-domain-host-mixed-20261004.Y66C6M9Q`; their files were not
overwritten or replaced. The earlier rejected kernel and failed integration
roots are also retained. No production promotion, numeric-format serving
expansion, fleet deployment or performance parity claim follows from this work.
