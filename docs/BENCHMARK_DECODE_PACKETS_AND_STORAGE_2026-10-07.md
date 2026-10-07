# Decode operand packets and shared storage experiments

Wider QK shared loads reduce instructions but do not reliably improve C16
decode. Removing sixteen unused shared bytes increases resident block capacity
from two to three, yet also does not improve C16 consistently. Neither change
is enabled in production. Long-history ordinary decode and one partitioned
control improve, so these results must remain shape-specific.

A diagnostic propagation bug is fixed: `String::replace` replaces only the
first occurrence. The original recipe rewrite changed the ordinary numeric
law but left the partitioned law unchanged; the extent rewrite likewise left
the partitioned shared reservation unchanged. Total replacements now update
both entries, regression tests assert both fields, and fresh full-chain
captures verify the corrected metadata. Original captures and failed command
journals are preserved with explicit limitations.

## Selected artifact and workload

Spark .179 uses GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. The executed composed decode module
is `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`,
source `dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6`.
It contains ordinary candidate 468 and eight partition partials plus merge.
Its selected law is `owned8-blockwise-f32-probability-v4`, KV tile 32, two
independent operand stages, BF16 storage and F32 probabilities.

The paired controls retain useful rows, captured launch envelope and history.
C16 uses rows 16, envelope 16, history 4095, grid 16 by 8 and block 64. The
long-history control uses rows 2, envelope 8, history 32767. The short control
uses rows 1, envelope 1, history 127. An additional extent control retains
rows 16 with envelope 32. Envelopes are not interchangeable with useful rows.

Five alternating pairs are measured per cell using the same independent
scalar oracle. The positive-effect decision requires every paired gain to
exceed 1%; independent medians are not sufficient. Counter replay times are
not unprofiled times or serving rates.

## Wider score packets

Packet widths two and four replace scalar BF16 shared reads with packed reads
and a bijective owner-local component traversal. Complete and tail score loops
are changed in both ordinary and partitioned entries. The arithmetic association
differs from the original dot traversal, so these have diagnostic numeric laws,
not a claim of general bitwise equivalence. All captured paired outputs happen
to be bitwise equal and pass the unchanged scalar oracle.

Corrected captures give the following median paired completion-time reductions.
A negative reduction means slower execution.

| Rows and envelope and history | Ordinary packet 2 | Ordinary packet 4 | Split and merge packet 2 | Split and merge packet 4 |
| --- | ---: | ---: | ---: | ---: |
| 16 and 16 and 4095 | 1.55%, inconclusive | 0.12%, inconclusive | 0.21%, inconclusive | -0.05%, regression |
| 2 and 8 and 32767 | 8.24%, improved | 6.72%, improved | -0.91%, regression | 1.08%, inconclusive |
| 1 and 1 and 127 | -0.08%, regression | 0.19%, inconclusive | 1.59%, inconclusive | 1.82%, improved |

The short split-chain gain is reproducible within these controls, but it does
not imply that serving should choose eight partitions for short history. The
long ordinary gain cannot be substituted for a partitioned serving result.
The original packet-four full chain was also inconclusive at C16 and long
history; correcting its label does not reveal a hidden serving gain.

A matched C16 packet-four counter capture has identical launch geometry:

| Counter | Baseline | Packet 4 |
| --- | ---: | ---: |
| Warp instructions | 41,500,160 | 38,248,960 |
| Registers per thread | 148 | 145 |
| Shared-memory resident block limit | 2 | 2 |
| Active occupancy | 8.07% | 7.98% |
| Eligible warps per scheduler cycle | 0.09 | 0.08 |
| Long-scoreboard cycles per active issue | 2.98 | 3.19 |
| Short-scoreboard cycles per active issue | 3.80 | 5.82 |
| MIO throttle cycles per active issue | 2.20 | 0.62 |
| Barrier cycles per active issue | 0.42 | 0.41 |
| Profile replay duration us | 1191.55 | 1199.39 |

Instructions fall 7.83%, but short dependency waits rise approximately 53% and
eligible issue does not improve. Scalar multiply/add counts are unchanged;
the savings are supporting load and conversion work. No register spilling or
source-correlated excessive shared wavefronts were recorded. These counters
support a tradeoff between load issue pressure and consumer dependencies, not
the claim that all load latency is solved by packing. A sampled waiting PC
alone does not identify the producer responsible for that wait.

The corrected packet-four CUBIN is byte-identical to the counter-captured
packet-four module: `31aa27591a2b57d28613179dae85408beedd740e17900708c0254f48404baf3d`.
Correcting comments and recipe labels does not invalidate that instruction
comparison. Diagnostic recipes remain unsuitable for signed serving admission;
the transformed source does not have a new production functional plan.

## Shared reservation boundary

The selected source retains both key and value validity in registers and has
no shared `tile_valid` or `stage_valid`. Its live shared extent is 32768 bytes
for staged BF16 operands plus 256 bytes for scores: 33024 bytes. The original
recipe conservatively reserves another sixteen bytes for validity.

The extent ablation changes only launch reservation, leaving the CUBIN, numeric
law and instruction count identical. The ordinary counter capture confirms:

| Counter | Reservation 33040 | Reservation 33024 |
| --- | ---: | ---: |
| Driver-inclusive shared bytes | 34064 | 34048 |
| Allocated shared bytes per block | 34176 | 34048 |
| Configured SM shared bytes | 102400 | 102400 |
| Shared-memory resident block limit | 2 | 3 |
| Active occupancy | 8.08% | 11.03% |
| Eligible warps per scheduler cycle | 0.09 | 0.10 |
| Short-scoreboard cycles per active issue | 3.81 | 6.08 |
| MIO throttle cycles per active issue | 2.18 | 4.67 |
| Warp instructions | 41,500,160 | 41,500,160 |

The capacity boundary is real, but higher residency increases shared pipeline
and dependency pressure. It does not produce a corresponding eligible-issue
increase. This falsifies the proposal that this capacity limit alone explains
the C16 gap. Driver overhead and allocation granularity belong to backend
resource feedback, not hardcoded portable IR constants.

| Rows and envelope and history | Ordinary extent reduction | Corrected split and merge extent reduction |
| --- | ---: | ---: |
| 16 and 16 and 4095 | -0.55%, regression | -0.44%, regression |
| 16 and 32 and 4095 | 4.23%, improved | -0.33%, regression |
| 2 and 8 and 32767 | -1.36%, regression | 2.55%, improved |
| 1 and 1 and 127 | -0.01%, regression | -0.38%, regression |

The corrected long-history split chain gains at least 1.05% in every pair,
with median gain 2.55%. All extent outputs are bitwise equal. This is a useful
candidate for measured shape selection, not a global default: other controls
lose. A production implementation must derive live storage from the physical
validity ownership proof, retain the conservative region where that proof is
absent, and propagate the resulting launch contract through AOT generation.
No source-string replacement belongs in that production pass.

## Remaining work and serving status

The previous exact serving attribution remains authoritative: decode attention
and merge account for roughly 47% of C16 client completion, while no-GPU time
is about 2.9%. These experiments target that cost but do not close it.
The remaining objective is to reduce ordered consumer dependency and shared
pipeline pressure while retaining enough independent work; maximizing
occupancy or minimizing instruction count in isolation is insufficient.
Ordinary, mixed-envelope and partition/merge paths require separate decisions.

The last verified serving result remains 225.600 output tok/s for 4096/64 C16.
Historical matched-workload vLLM/SGLang results are 243.03/241.85 tok/s;
LunaFlux's completion overhead against those results is roughly 7.7%/7.2%.
Neither reference nor end-to-end serving was rerun here.
The token-parity gate in the prior common-route campaign remains open.

## Reproduction and preserved evidence

Automation is MoonBit-only: `decode_score_packets_20261007.mbtx`,
`decode_shared_extent_20261007.mbtx`, the paired mode in
`decode_consumer_window_20261007.mbtx`, `finish_decode_packets_20261007.mbtx`
and `verify_decode_packets_20261007.mbtx`. Both-field propagation regressions
and packet ownership tests pass. No model branch, request-path profiler,
runtime JIT or production numeric law is added.

GPU jobs were serialized. User-systemd diagnostics were capped at 8 GiB,
zero process swap and 900 seconds. Counter containers also had an 8 GiB limit.
Paired cells checked a 32 GiB host reserve before and after; packet and counter
runs additionally sampled reserve every 500 ms. Original failed chain startup
and corrected-build argument-order failures are retained in journals, not
silently removed. No soak, deployment or reference server was changed.

All six archives and every `FILES.sha256` entry were verified locally under
`/tmp/lunaflux-decode-packets-evidence-20261007.aU0AfVXY`.

| Evidence | Remote directory suffix | Archive SHA256 |
| --- | --- | --- |
| Original packets | `lunaflux-decode-score-packets-20261007.j7vv8uX8` | `515d6f13540c607805503e5e41bd06396920a53f7bbfe4b61390efa628bae373` |
| Packet counters | `lunaflux-score-packet-counters-20261007.rzARzf8h` | `c22ae319cb611efabf222f5c55aa2a3dcbc9a32e040b787eb2af77e908919d28` |
| Ordinary extent | `lunaflux-decode-shared-extent-20261007.3fd7MxqT` | `60bf3ec6bf23af629a7bf0058f56ea32584f5d5d73575f9330a2095101bd3ee2` |
| Extent counters | `lunaflux-shared-extent-counters-20261007.8IFLhL1Y` | `d01cc5b0c20239649a7317545335b997036d1470b13de46deec47647f0129805` |
| Corrected packets | `lunaflux-decode-packet-propagation-20261007.AvW4s47L` | `622e392fec012a1689333d8196a9e9599b6f28ba1e9dd90be9bd9b4b248ffb69` |
| Corrected extent chain | `lunaflux-decode-extent-chain-20261007.oVAKqwnS` | `5879a559e46a1baaa5304d00f3356983e1ec1fa9f2268b9100895f077ff062a0` |

Remote directories are under `/home/wlc004s/`; archives append
`.verified.tar.gz`. Original selected source, module, recipe and probe are
retained as frozen controls alongside measurements. The packet and extent
alternatives remain offline experiments, not evidence of universal parity.
