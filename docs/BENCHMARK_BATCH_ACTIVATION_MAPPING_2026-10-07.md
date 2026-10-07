# Batched activation differences in the selected Qwen runtime

## Result

The current 32,512-token fixture reproduces a solo-versus-paired token
difference. In the captured first layer, the first changed live boundary is
the V portion of the fused QKV ingress output: two BF16 components differ at
sample 1, by at most 0.00006103515625. Q and K after normalization and RoPE
match at that step. Small attention-output and downstream differences appear
at samples 2 and 3. The first differing output token is sample 6 for request
row 0; all eight tokens for request row 1 match.

This narrows a numerical propagation question; it does not establish a
numerical-contract violation, a memory race, or the cause of the performance
gap. The capture includes only the first five sampled positions and the first
two layers' selected boundaries. Earlier cached KV and all later layers are
not compared, so the observed first-layer differences are not sufficient to
attribute the eventual token choice to one producer.

An actual diagnostic bug is fixed: fused attention rebinds its semantic input
to a separate physical sidecar. The first probe read the old activation-arena
region and reported thousands of differences that were not consumed by the
attention kernels. Corrected readback follows the actual prepared allocation
and offset. The original run remains preserved and its corrected analysis
marks that cut non-comparable.

No production algorithm, numerical tolerance, selected module or policy is
changed. The [October 6 end-to-end measurements](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md)
remain the latest performance results: 4096/64/C16 is 225.13 tok/s versus
243.03/241.85 for vLLM/SGLang; 32512/64/C2 is 16.95 versus 17.58/18.85.

## Exact workload and diagnostic execution

The frozen runtime is
`/home/wlc004s/lunaflux-mixed-capture-repair-20261006.sQMel69Z`.
The saved input bodies are the two 32,512-token requests under
`/home/wlc004s/lunaflux-mixed-final-benchmark-20261006.Xzq0NMMd/r0-luna-full/requests`.
Only the requested output length changes from 64 to 8. Each fresh server runs
solo row 0, solo row 1 and paired rows 0/1, for four requests and 32 output
tokens. Both captures produce the same complete token vectors.

The worker is a diagnostic build of that frozen runtime. Its actual selected
module roots stay pinned to the frozen deployment; synchronous activation and
logit readbacks require eager execution. Consequently neither elapsed times
nor graph markers in this capture represent uninstrumented serving performance.

Spark .179 uses GB10 sm121, GPU UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Preparation is bounded to 16 GiB
with no process swap and a 1,200-second limit. Serving is bounded to 48 GiB,
the bridge to 2 GiB, with no process swap and a 900-second limit. The launcher
checks GPU idleness and preserves a 32 GiB MemAvailable reserve. The corrected
run's minimum recorded MemAvailable is 104,552,316 KiB. Both supervisors drain,
acknowledge cancellation, exit zero and close their children. The terminal GPU
has no compute process.

## Physical storage correction

The model plan describes activation 4 as the rotated QKV input to operation 5.
However, `prepare_qwen_fused_attention_launch` replaces that argument with the
fused sidecar at offset zero. The partitioned-prefill and split-decode argument
builders use the same sidecar. Reading the corresponding logical region in
the shared activation arena therefore does not observe this physical input.

The correction is limited to the offline fixture controller. It verifies a
live sidecar exists and reads its row using the semantic row stride and the
actual packed query-row offset. Other captured inputs still use their prepared
activation-arena regions. This explicit fixture mapping is not a model-specific
branch added to the compiler or production token path.

The Qwen ingress producer writes the complete packed region, with Q occupying
components 0–2047, K 2048–3071 and V 3072–4095. In the corrected sample-1
comparison, only components 3167 and 3402 change. The large apparent changes
in the original semantic-arena readback must not be interpreted as corrupted
QKV values.

Request IDs and generations come from the worker trace. Client responses join
to those identities using complete eight-token vectors within each sequential
wave, not arrival order or a reused batch-row index. Ambiguous joins fail
explicitly. Repeated physical launches retain their occurrence counts; a
different number of launches is compared as one input cut only when the input
words are proven identical across every occurrence on each side. Fused
operation 3 has no surviving standalone launch in this capture.

## Corrected first layer comparisons

The table compares solo row 0 with the same request in the paired wave. Solo
decode has one packed token; the paired wave has 2,048 packed tokens while the
other request continues prefill. All captured QKV inputs match. At sample 0,
every captured live boundary matches despite the different prefill-tail sizes
of 1,792 versus 2,048 tokens.

| Sample | Live boundary | Changed BF16 components | Maximum absolute difference |
| --- | --- | ---: | ---: |
| 1 | Rotated QKV sidecar, V only | 2 / 4096 | 0.00006103515625 |
| 1 | Attention output | 0 / 2048 | 0 |
| 2 | Rotated QKV sidecar, V only | 2 / 4096 | 0.000244140625 |
| 2 | Attention output | 2 / 2048 | 0.00006103515625 |
| 2 | MLP normalized input | 6 / 1024 | 0.001953125 |
| 2 | Second-layer QKV input | 175 / 1024 | 0.001953125 |
| 3 | Rotated QKV sidecar, V only | 2 / 4096 | 0.000244140625 |
| 3 | Attention output | 2 / 2048 | 0.000244140625 |
| 3 | MLP normalized input | 35 / 1024 | 0.00390625 |
| 3 | Second-layer QKV input | 266 / 1024 | 0.001953125 |
| 4 | Rotated QKV sidecar, V only | 1 / 4096 | 0.000000476837158203125 |
| 4 | Other captured live boundaries | 0 | 0 |

Request row 1's captured live boundaries all match. There are 140 activation
records, including repeated physical launches. The independent host argmax
agrees with device sampling for all 32 outputs, and five top-two BF16 logit
ties occur. This checks selection from the captured logits, not model quality
or parity with reference engines.

Projection schedule or reduction-order sensitivity is a plausible explanation
for the V differences: the projection input matches, and V bypasses Q/K
normalization. This remains an inference until exact input/weight replay checks
the selected projection alternatives against their declared numerical law.
The attention output comparison also needs the earlier historical cache before
it can distinguish new V sensitivity from previously accumulated differences.

## Fixed tooling and next decision

The controller now creates the campaign directory before the authoritative
launcher writes its manifest, reads the fused sidecar, and preserves the exact
diagnostic overlays and binaries. The report rejects dead semantic storage as
a causal numerical comparison and preserves physical sublaunch distinctions.
Both helpers pass warning-denied native checks and six regression tests pass.

The next numerical experiment should replay the first changed V projection
with exact input and weights, rather than force all shapes onto one slower
schedule for bitwise equality. Any production change must satisfy the actual
numeric contract and preserve the generic schedule/ownership/lowering layers.
The separate performance priority remains the tensor and fragment dependency
chain identified by the [selected prefill instruction samples](BENCHMARK_PREFILL_OPERAND_PIPELINE_2026-10-07.md).
This readback does not replace that counter diagnosis or demonstrate a speedup.

## Preserved runs

- Original semantic-arena capture:
  `/home/wlc004s/lunaflux-activation-latest-20261007.txYa7WCh`.
  Use `activation-comparison-corrected.json`; its operation-5 cut is invalid.
- Corrected physical-sidecar capture:
  `/home/wlc004s/lunaflux-activation-latest-20261007.GVxvruur`.
  Use `activation-comparison.json` with `attention_input_physical_sidecar=true`.

The original launch failed before GPU execution because the campaign directory
was missing. Its command and error remain alongside the successful startup
repair. A later archive-preparation path-flattening error occurred before any
overlay content was written; the failed finalizers remain preserved alongside
the corrected finalizer. Neither failure replaces or relabels a GPU result.

Archive SHA-256 values are respectively
`df5ce949d21e03686869013a74a5f03208eb8d64ea6e820617aecbcda62e7079`
and `5c33e95462ba9dffea1aecb6be1a7e4da1651b688b0a57a07104fa51c676cb9d`.
Distinct downloads under
`/tmp/lunaflux-activation-latest-20261007.1eSqayXk` match those hashes.
Both extracted `MEASUREMENT.sha256` manifests verify locally.
