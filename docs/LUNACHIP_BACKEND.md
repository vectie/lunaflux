# LunaChip experimental backend

On PBlaze3 images with the 1 KiB row reader, set
`PBLAZE3_ALIGN_MATRIX_1K=1` for model preparation and subsequent runs.
LunaFlux places each BF16 matrix on a 1 KiB boundary. It uploads the
original tensor bytes and includes the destination addresses in the
resident-weight identity. Switching between this layout and the compact
layout requires a fresh resident installation. Both use the same checkpoint.

2026-10-29. This backend consumes the published PBlaze3 device contract over
serialized data. It does not import LunaChip, invoke CUDA, or install a serving
backend implicitly. LunaFlux owns model loading, tokenizer, KV ownership,
token scheduling and sampling. LunaChip owns the reference compute design,
device ABI and hardware/simulator evidence.

The first target is Qwen3-0.6B, batch one, 2048 total tokens, BF16 row-major
matrix storage and FP32 activations/KV. Existing Qwen3 metadata supplies model
geometry and source-tensor ordinals. Its BF16 CUDA operation schema is **not**
the FPGA numerical contract: ordered FP32 multiply/add and approximate
nonlinear functions have their own profile. The existing Qwen3 spec remains
unchanged; no global relaxation of its BF16 KV contract is permitted.

Preparation allocates the tied embedding once, all 28 layers, normalization
copies, per-layer K/V, reusable workspace and a reserve. Every region is checked
against the 2 GiB aperture with wide arithmetic. Model association labels are
consumed as labels; no whole-checkpoint hashing is added. Tensor transfer must
reuse the existing Qwen3 weight binder and bounded tensor ranges.

The first compiler slice accepts validated LunaTile programs containing whole
contiguous one-dimensional FP32 copies and elementwise add/multiply. Register
tensors map to disjoint scratch ranges, global operands require explicit DDR
bindings, and uninitialized reads are rejected. Unsupported precision,
MMA/reduction/pipeline operations, loops other than one singleton iteration,
and partial views fail explicitly;
there is no CUDA or CPU fallback. This intentionally avoids promising a
numerical order for generic reductions that the source language has not fixed.
Programs split at 2047 operations plus HALT, preserving scratch across launches.

Only experimental startup/offline preparation and diagnostic execution are
admitted. The follow-up below implements full-model Qwen lowering and DDR
attention, with independent host logits/token agreement and two-position
physical full-model generation. Long-context physical validation and real
PCIe execution remain open gates. The memory layout does not make the
existing production BF16 executor compatible with this target. Reusable launch
tracking is transport-neutral; real I/O must enforce bounded completion and
exclusive device ownership. Uncertain START/timeout requires recovery, never
automatic retry.

Validation covers contract/version/capacity rejection, complete model layout,
tied storage, final-context KV addresses, capacity exhaustion, actual descriptor
bytes, splitting, unsupported LunaTile cases, and stale/failed launches.
A serialized LunaFlux fixture is executed by LunaChip's existing LunaWave
core/controller/AXI bench. This is digital integration evidence; whole-model
numerical qualification and physical execution remain separate.

## Implemented preparation and validation

- `device/lunachip`: strict serialized ABI admission, checked copy/add/multiply/
  load/store and BF16 GEMV descriptors, immutable HALT-terminated chunks and exclusive launch
  receipt tracking. The transport must read the sequence around each status
  snapshot and invalidate its owner after timeout or uncertain START.
- `compiler/lunachip_tile`: lowers the admitted LunaTile subset directly from
  read-only canonical views. No serialized CUDA source or LunaChip IR is parsed.
- `model/qwen3_lunachip`: complete 310-tensor placement derived from existing
  Qwen3 metadata. Source BF16 weights occupy 1,192,099,840 bytes (1136.875 MiB),
  FP32 normalization copies 262,144 bytes, FP32 KV 469,762,048 bytes (448 MiB).
  With 64 MiB workspace and 64 MiB reserve, 351,141,888 bytes (334.875 MiB)
  remain. The full-Qwen diagnostic plan uses 615936 bytes of workspace;
  the 64 MiB reservation leaves room for future admitted plans.
- `integration/lunachip_qwen`: joins the existing validated Qwen safetensors
  binder to destination regions. Streaming chunks are at most 1 MiB and retain
  whole BF16 elements. Source offsets are relative to the safetensors payload;
  add the inspected header/data start when issuing file reads. Norm expansion
  writes FP32 copies at the separate destination, with twice the source offset.
- `cmd/lunachip_prepare`: consumes the actual Qwen config and a supplied model
  association label, emits a JSON placement plan and opens no device.
- `cmd/lunachip_fixture`: developer-only compiler fixture export for LunaChip's
  digital integration bench. It is not part of the serving runtime.

From the two sibling checkouts:

```sh
cd /Users/kq/Workspace/lunachip
moon run --target native cmd/pblaze3 -- contract > /tmp/pblaze3-contract.json
cd /Users/kq/Workspace/lunaflux
moon run --target native cmd/lunachip_prepare -- /tmp/pblaze3-contract.json /absolute/model/config.json MODEL_ASSOCIATION_LABEL_64_HEX
moon run --target native cmd/lunachip_fixture -- /tmp/pblaze3-contract.json > /tmp/pblaze3-lunatile.json
cd /Users/kq/Workspace/lunachip
moon run --target native cmd/pblaze3_adapter -- /tmp/pblaze3-lunatile.json
```

Use fresh output paths when retaining a run. The actual Qwen config used in the
2026-10-29 preparation is the existing pinned source config in the LunaChip
checkout. Its existing config identity was supplied as an association label;
the command does not authenticate checkpoint contents.

The cross-repository fixture passed in LunaWave: six actual descriptors,
1050 clock edges, eight AXI read beats, four write beats, four bit-exact FP32
outputs and two untouched neighboring words. It loads at the end of the
2 GiB aperture and writes at an address offset within a 16-byte AXI beat,
with seeded backpressure. This does not execute Qwen layers or model generation.
The same exported canonical RTL and stimulus pass strict lint and replay under
the repository-pinned Verilator 5.050: every output row at all 1050 edges is
byte-identical. The 10 adapter/placement tests, 148 existing LunaTile tests and
12 LunaChip host tests pass (170 total).

Full strict native checks currently encounter pre-existing MoonBit migration
warnings in LunaFlux dependencies. Targeted validation uses the repository's
existing migration exclusions `-20-25-29-35-79-92`; they do not authorize ignoring
new adapter warnings. The LunaChip side is checked without those exclusions.
The complete LunaFlux native check passes with those existing exclusions.
This preparation is not a completed Phase 8 or production release gate; the
entire test suites, physical inference and model numerical gates are not claimed.

## BF16 GEMV integration, 2026-10-29

`compiler/lunachip_tile.OrderedBf16Gemv::new(weights~, activation~, output~)`
accepts existing LunaTile tensor views with explicit mixed-precision semantics.
Weights are contiguous global BF16 `[N,K]`, activations global FP32 `[K]`,
output global FP32 `[N]`. `N` is 1–4096, `K` is 64–4096 and divisible by 64.
Weights require 128-byte alignment; vectors require 4-byte alignment. Offsets
are bytes added to the caller's allocation base; all three resolved ranges
must be disjoint and contained in the 2 GiB aperture. Transposes, packed
stripes, noncontiguous layouts, negative offsets and implicit dtype conversion
are rejected. Raw view constructors are not assumed to validate these facts.

`operation.lower(device, bindings)` returns immutable LOAD/opcode-4/STORE/HALT
bytes. Scratch input begins at word 0 and output at 4096, so all admitted shapes
are disjoint. Arithmetic is BF16→FP32 exact expansion, separately rounded RNE
FP32 multiply, then increasing-K FP32 addition from +0. It is not FMA and does
not round activations or KV to BF16. See ADR-0020 for the generic MMA boundary.

`integration/lunachip_qwen.prepare_projection(device, layout, ordinal,
first_row, rows, activation_address, output_address)` selects a row tile from
the existing Qwen matrix allocation. Both vectors must fit inside workspace;
weights, normalization copies, KV and reserve cannot be used as its output.
Ordinal zero reuses the tied embedding/output matrix, including vocabulary
tail tiles. Preparation does not upload a checkpoint or execute a whole layer.

The developer fixture command accepts a `bf16` argument:

```sh
moon run --target native cmd/lunachip_fixture -- /tmp/pblaze3-contract.json bf16
```

It exports 16×128 parallel and 3×64 scalar GEMV programs, BF16 weight bytes,
FP32 activations and independent ordered host results. Cases distinguish
non-fused rounding, ordered cancellation and accidental BF16 activation
rounding, and cover BF16 subnormals, infinities, NaNs and upper DDR addresses.
The LunaChip adapter bench replays these descriptors against actual canonical
hardware IR with seeded AXI backpressure. Digital evidence and results are
recorded in the sibling LunaChip BF16 evidence directory; physical inference
is still unqualified.

The associated LunaFlux audit found and fixed a generic CUDA reduction defect:
`Maximum` used `-FLT_MAX` as its initial value, changing an all-`-Inf` input
to a finite answer. It now emits `-INFINITY`; a source regression fails before
the fix and passes after. The existing generic MMA contract remains unchanged.
This is a focused audit of this integration's dependencies, not a certification
of all LunaFlux numerical paths or a new physical CUDA qualification.

The BF16 fixture passed the actual LunaWave composition: seven descriptors,
6779 clock edges, 472 AXI read beats, 19 write beats, 19 bit-exact FP32 results
and four preserved guard words. Pinned Verilator 5.050 passed strict lint,
build and replay, matching every output port at every edge. The 16 adapter,
19 LunaTile IR, two Qwen spec, ten Qwen weight-binding and 12 LunaChip host
tests pass (59 distinct tests). These targeted tests include compilation of
row tiles for all 197 Qwen matrices and the tied vocabulary's final row.
The LunaChip native check passes without warning exclusions; the LunaFlux
native check passes with the existing migration exclusions documented above.
No new warnings were found in the new BF16 implementation/test files.

## Full Qwen diagnostic follow-up (2026-10-29)

`prepare_decode_step(device, layout, position)` now emits all 28 layers and the
final RMSNorm/tied output head. It supports the full 2048-token layout with
DDR-backed FP32 KV. Each layer's descriptors split into HALT-terminated chunks
while scratch ownership persists. Attention scores use bounded key tiles;
value accumulation retains original position order instead of adding tile sums.

`cmd/lunachip_qwen` accepts `prepare`, `simulate`, and `physical`, followed by
contract path, model directory, a 64-hex association label, fresh output
directory, raw prompt and generated-token limit. Physical mode adds an SSH
loopback endpoint and a combined operator-qualification JSON fixture to the
separately qualified vendor JTAG bridge. Model files
are `config.json`, `tokenizer.json`, and `model.safetensors`. The physical mode
checks the device contract and source header, verifies every uploaded tensor
range by actual DDR readback, executes the prepared programs in order, and
compares full physical logits with both an ISA interpreter and independent
Qwen equations. Before model upload it verifies 42 exact physical operator
results, 18 guards and a cross-page asymmetric transport pattern. It never
substitutes the reference for device operations. Terminal token IDs 151645
and 151643 stop greedy generation.

Real checkpoint software qualification used official Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`. `Hello` produced ` Answer!`; the
five-token prompt `The capital of France is` produced ` Paris.`. All eight
evaluated positions matched greedy token choices across all 151936 logits.
The largest observed absolute logit difference was 0.00009071826934814453.
The independent reference uses libm nonlinearities and ordered FP32 products/
sums; the descriptor interpreter uses the published FPGA approximations.
These two runs are host results. A subsequent physical run completed on
2026-10-30 00:11 Asia/Shanghai through JTAG with `Hello` → ` Answer!`:
all 1,192,099,840 selected BF16 weight bytes passed DDR readback, and both
151936-word physical logit vectors match the ISA interpreter exactly. The
independent reference chooses the same tokens, with the same errors as the
`Hello` host run. All 28 layers, normalization, attention, softmax, SiLU,
residuals and tied output head execute on FPGA; the host owns tokenization,
embedding lookup, RoPE input constants and greedy sampling. This is a complete
prefill position plus a KV-backed decode, not long-context or serving admission.

The diagnostic executable's allocations, source reads, polling, and reference
checks are outside production serving claims. The CLI limits runs to 64 total
tokens / 16 generated tokens. See ADR-0021 and the sibling LunaChip
`docs/PBLAZE3_FULL_QWEN.md` for physical acceptance status and retained evidence.

Host coverage includes 19 adapter/placement/complete-plan tests and five
diagnostic tests. The latter exercise stale/malformed replies, exact byte order,
nonfinite logits/EOS handling, and a 607744-byte full-vocabulary read whose
ASCII response is split into odd-sized fragments. The mock uses a linear
string builder: the dependency's generic hexadecimal helper prepends immutable
strings and blocks the test scheduler on megabyte fixtures. The actual vendor
bridge uses Tcl's binary encoder; this finding does not change the qualified
physical client's receive path or its running executable.

Resident performance preparation (2026-10-30): `cmd/lunachip_qwen benchmark`
uses the optional LunaChip controller capability at `0x90`. All prepared
programs are uploaded once into a disjoint tail of the workspace DDR region;
each launch sets the DDR base/count and the FPGA stages its own instruction
RAM. No weights or descriptor bytes are uploaded in a timed token step.
The vendor bridge only implements checked transport/launch/wait/snapshot;
LunaFlux retains all model planning, token ordering, KV ownership and sampling.

After one warm-up, three trials use a monotonic timer. Each token retains the
accepted START sequence, full 64-bit core cycle count, loader cycle count and
transport request bounds. Timed work includes embedding/RoPE input transfer,
launch/polling, all FPGA computation, full 151936-logit readback and greedy
sampling. Evidence writes and every ISA/equation reference operation happen
after all trials finish. Full logits and token histories must agree across
trials and with the references before a benchmark PASS is published. This is
a diagnostic JTAG benchmark, not production allocation/latency admission.
The new LunaChip bitstream was generated with positive setup/hold/pulse timing
on 2026-10-30. The user requested ending this phase before board programming.
Physical performance and 30 MHz TCK qualification therefore remain pending;
no new tokens/s result is claimed. The Linux release client is built and its
seven targeted tests pass. Keep the subsequent physical run in fresh evidence.

The November 10 `benchmark-fast` mode uses the FPGA's streaming FP32 argmax,
packs layer programs into instruction windows and preloads per-position RoPE
constants. It arms the output range, writes the input embedding, dispatches
the windows and reads the selected token. Full per-position logits remain in
disjoint DDR slots until the trial's timer stops. The host then reads the
vectors and checks them against the ISA and equation reference after all three
trials. Run `simulate-fast` to execute the same packed descriptors offline.
The legacy `benchmark` mode supports the October resident image.

On November 11, `benchmark-fast` completed three physical batch-one trials
using 30 MHz JTAG and the 25 MHz PBlaze3 core. Starting from raw `Hello`,
each trial emitted four tokens. End-to-end generation measured 12 tokens in
658.930111 s (0.01821134 tokens/s). Decode after the first token measured
9 tokens in 495.341000 s (0.01816930 tokens/s). The transport trace confirmed
zero timed weight or program uploads. All three trials' full logit vectors
matched the ISA replay byte for byte at all four positions. Reference execution
and full logit reads occurred outside the timers. Per-token cycles and raw
results are in the LunaChip campaign `artifacts/pblaze3-fast-physical-20261111-a1`.

With the 256-bit streaming PBlaze3 image the same LunaFlux executable and
checkpoint produced the same twelve full logit vectors byte for byte. The
three four-token trials took 54.412 s in total, or **0.22054 tokens/s**.
Decode after the first token reached **0.21481 tokens/s**. The client uses
the same device contract, durable program addresses and out-of-timer reference
checks. The physical counters measured 91,916,158 core cycles per output on
average, plus 5,791,245 instruction-loader cycles. The comparison and all
twelve samples are in the LunaChip campaign
`artifacts/pblaze3-bandwidth-20261111-a1`.

The 40 MHz vector image generated twelve tokens in **33.55468 s**
(0.35763 tokens/s), with 0.34696 tokens/s for the nine decode positions.
The same row-major checkpoint on the 25 MHz vector image took 46.61533 s
(0.25743 tokens/s). Both used a 30 MHz JTAG cable. The transport handshake
returns the routed core frequency for the 40 MHz image. A legacy scalar
reply denotes the 25 MHz image. `benchmark-fast` also accepts an optional
final argument pointing to `diagnostic-routed.txt` to verify the core,
instruction loader and AXI clock against the bridge. The clock conversion
and reference checks run outside the timed generation loop. The complete
physical comparison is in the LunaChip campaign
`artifacts/pblaze3-weight-clock40-20261111-a1/comparison.json`.

## FPGA-driven resident segment dispatch

`lunachip_qwen benchmark-chain` accepts the same arguments as
`benchmark-fast`. It checks the chain capability and packs each position's
resident ranges into one base/count command. The descriptor bytes, constants,
embedding updates, persistent KV and four-byte argmax read use the existing
fast plan. The 64-lane hardware stages each 2,048-descriptor segment and
executes its HALT before loading the next segment. LunaFlux checks the accepted
generation, total retired instructions, completed segments and cycle totals.

The paired benchmark uploads the model and programs once. The same routed
image first runs the segmented control and then the chain, with independent
warm-ups and three timed trials per mode. Each trial reads all logits after its
timer stops. The client compares both modes byte for byte, replays the ISA,
and runs the independent equation reference. A sample's dispatch overhead is
its dispatch duration minus the clock-derived core and loader time.

PBlaze3 PBR1 benchmark: `lunachip_qwen benchmark-program-bursts` accepts the
same arguments as `benchmark-chain`. It runs both modes on one routed image
and one resident upload. Each mode does one warm-up followed by three trials,
one CHN1 START per generated token. The five loader-counter deltas are taken
outside timing; the paired control must match every candidate FP32 logit
before the independent model validation. Earlier CHN1 images retain the
`benchmark-chain` entry point. After the timed comparison and independent
validation, the PBR1 client records the same four positions by individual
segment in `prefetch-segments.json`, checking the split run's logits against
the chained run. The qualified 40 MHz board averaged 1.43371 tokens/s and
302,161 loader cycles per output (same-image scalar: 1.17269 and 6,094,121).

## PBlaze3 resident inference across client processes

On a continuously powered board, start the JTAG bridge in `attach` mode. Use `generate-warm` for ordinary generation and `benchmark-warm` for repeated timed generation with post-timing logit validation. Both accept the same arguments as `generate-resident`. Each checks the DDR health latch and cached model identity before its first board write. A missing or different model ends the request immediately. Use `generate-resident` to allow a cold weight upload. Reprogramming the FPGA clears the residency latch.

On the SFU1 image the warm benchmark runs one warm-up and three trials with burst program loads, streaming unary operations and one START per output. The updated JTAG bridge measured 1.734 tokens/s for the raw prompt `Hello` and four outputs, compared with 1.639 tokens/s using the earlier bridge on the same image. The LunaChip campaign `artifacts/pblaze3-warm-dispatch-20261115-a1/RUN.md` has the installed Linux paths and invocation.

For a fresh prompt on the SFU1 image, use `lunachip_qwen generate-resident CONTRACT MODEL_ROOT ASSOCIATION_LABEL_64_HEX NEW_OUTPUT PROMPT NEW_TOKENS LOOPBACK_ENDPOINT QUALIFICATION_JSON [ROUTED_CLOCK_REPORT]`. The bridge accepts `program` for the initial configuration and `attach` to reopen the programmed board. Start one bridge per client. Use an exclusive connection and a fresh output directory. The client runs one greedy generation and writes the text, token IDs and timings to `generation.json`. It downloads the selected token word at each position. Provide the conversation prefix for a new process to reconstruct its KV state. The current diagnostic accepts up to 64 total prompt and requested output tokens, including at most 16 new tokens per call.

`residency.json` records the actual checkpoint SHA-256, weight and program cache decisions, uploaded BF16 payload bytes, weight readback bytes and cold-upload time. The client and bridge hash their local checkpoint files. A warm attach reads the small board identity record. A new prompt with the same shape reuses the program bytes. A changed context layout may require a new program upload. The cold path uses write-only BF16 checkpoint transfers by default. Set `PBLAZE3_VERIFY_WEIGHTS=1` to request full BF16 readback on a cold upload. Program and expanded-norm writes retain their ordinary checked transfers. The benchmark entry point `benchmark-stream-unary` uses the same residency protocol and checks all logits after timing. Both modes assume exclusive board ownership. Clear the residency latch before using an external writer that accesses cached ranges.
## Ordered Qwen decode programs for PBlaze3

The Qwen3-0.6B adapter joins physically adjacent query, key and value weights into one 4,096-row BF16 matrix instruction. Each row retains its original FP32 accumulation order and scratch destination. The ISA simulator resolves the same consecutive DDR range across the three source tensors even when their safetensors file offsets differ.

The first decoder layer builds the position's cosine and signed-sine tables in retained scratch above the historical-key tile. One pair of FP32 vector multiplications prepares all 24 heads for RoPE. Independent RMSNorm scales share contiguous scratch slots, while each head still reduces its own 128 values in order. For attention, complete pairs of query heads share a 4,096-word score area. The tile fits all sixteen heads at contexts up to 256 and one pair at context 2,048. Each head retains its own maximum, ordered sum and normalization factor. The output logits use the same FP32 format and vocabulary order.
