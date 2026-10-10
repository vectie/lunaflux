# Independent GLM / DeepSeek checkpoint correctness — 2026-10-11

Internal ordinary/speculative agreement and successful resource cleanup are
not independent numerical correctness. This work keeps that distinction and
does not add production authentication, hashing, TLS or release prerequisites.

## Matching reference discovered

Both Spark `.178` and `.179` already contain
`ghcr.nju.edu.cn/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`.
Despite the image tag, its installed vLLM source contains both `Glm5Next` and
DeepSeek-V4 implementations. The older January SGLang image and local source
trees are not matching references for these architectures.

The new offline MoonBit runner is
`scripts/run-independent-checkpoint-reference.mbtx`. It mounts the original
staged checkpoints read-only, uses two tensor-parallel ranks, literal input
IDs, an eager 256-token context, one sequence and 512 MiB KV allocation.
Each container has a 96-GiB/no-swap limit; no GPU workloads overlap these
references. These are correctness experiments, not comparative throughput.

GLM's original configuration declares FP4 weights and **null input activation
quantization**. The runner translates that to the reference's `W4A16_NVFP4`
mode, preserving the original exclusion list. It does not substitute EXL3
weights or silently introduce FP4 activation quantization. `ABLIT=0` disables
the image's optional load-time weight edits; reasoning stop suppression is
also disabled. No checkpoint file is changed.

## Execution status

The first GLM attempt, `/tmp/lunaflux-independent-glm-20261011-v1`, failed
before weight loading. The concrete cause was Gloo advertising loopback
addresses between two machines, not a numerical discrepancy or OOM. The
runner now explicitly selects their shared `enP7s7` interface for Gloo/NCCL.
The second attempt retains its own commands and results under
`/tmp/lunaflux-independent-glm-20261011-v2`.

Independent full-model GLM/DeepSeek comparisons are still pending; model
architecture recognition or a running server is not a correctness pass.
Reference requests must use identical literal IDs to the native captures,
retain token/log-probability outputs, and distinguish greedy agreement from
teacher-forced numerical comparison. Existing native cancellation and
retained-weight results remain valid in their narrower scope.

## MiniMax follow-up

The existing independent fifty-layer encoder corpus remains the starting
point, including its larger long-input and digit-row differences. Matched
operator tests already show exact rotary application and very small projection
differences on the original three-token caption. Native arithmetic can be
closer to the wider F64 probe than library BF16 arithmetic; bitwise imitation
of one reduced-precision library path is not automatically a numerical fix.

The separate reference driver now exposes CPU-only `rotary-tables` to check
the F32 frequency/angle and BF16 cosine/sine tables across wider positions.
This isolates table generation from propagated hidden-state differences and
does not add a reference dependency to serving.

The production pure table planner is exercised separately by
`benchmarks/minimax_rotary_probe`. At all positions 0–139, both its cosine
and sine tables match the independently generated CPU ATen tables bitwise:
17,920/17,920 BF16 values in each, with zero absolute and relative error.
This rules out the table-generation hypothesis for the measured 140-token
text case; it does not claim GPU trigonometric equivalence or resolve the
propagated encoder differences. Results are in
`/tmp/lunaflux-minimax-native-rotary-20261011-v1` and
`/tmp/lunaflux-minimax-aten-reference-rotary-20261011-v1`.

Inspection of the actual installed upstream encoder also confirms that it
uses three separate Q/K/V GEMMs and separate gate/up GEMMs despite packed
parameter storage. The independent encoder already follows that operation
order. Switching its oracle to packed GEMMs would change the reference,
not fix the native implementation.

## GLM reference arithmetic caveat

The installed reference warns that gate/up global FP4 scales differ. Its
`ModelOptNvFp4FusedMoE.process_weights_after_loading` then selects
`w13_weight_scale_2[:, 0]` for both projections. LunaFlux retains the original
separate gate/up/down scale planes. Consequently an unmodified reference
completion is useful as a differential but cannot certify exact original-
checkpoint arithmetic. Its sparse MLA backend also selects FP8 KV storage.
These differences must be reported, not silently treated as native bugs or
fixed by altering the original checkpoint.
