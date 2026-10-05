# AKO dual-Spark batch/chunk numerical isolation

## Outcome

Both Sparks were used independently, without competing GPU workloads. This
round completed 20 serving replay waves (32 request vectors) on .179 and 16
cross-phase attention checks on .178. It introduces diagnostic automation,
not a production kernel or scheduling change.

**The mismatch is not exclusively a concurrency problem.** The second saved
32,512-token prompt produces different greedy vectors under 2K and 8K chunks
even when served alone. Each solo result is reproducible within its own arm.
The first saved prompt remains equal across solo arms. Previous C1 parity for
that first prompt does not establish parity for all prompts.

The isolated prefill/decode comparison also establishes that the selected
phase laws are not bitwise equivalent on identical synthetic query/KV inputs.
This does **not** prove that attention alone caused the full-model mismatch.
Actual layer intermediates and logit margins are still needed to separate
projection geometry, cached-KV rounding and attention phase selection.

No concurrent numerical-equivalence admission or new performance win is
claimed. Existing serving defaults and immutable deployments are unchanged.

## Serving replay on .179

Host: Spark GB10, sm121, GPU UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.

Source-bound prepared deployment:
`/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`.
Both arms reuse its same 8K-capacity AOT bundle and binaries; only the
scheduling chunk ceiling differs. Runtime bundle SHA-256:
`dfb6e35a8940c4e060b8c6d7397f11bbb86360d03b4e6a78dcd72144e4973ac9`.

The exact two saved bodies are copied from
`e2e-0-c2048/r0-luna-full/requests/i32512-o64-c2-varied-t1-r{0,1}.body.json`.
Each contains 32,512 input token IDs and requests 64 outputs. Tokenization,
model, prefix-reuse policy and numerical tolerances are not changed.

For each arm, one fresh server executes two repeats of five modes:

1. Prompt 0 alone.
2. Prompt 1 alone.
3. Both submitted together, order 0 then 1.
4. Both submitted together, order 1 then 0.
5. Prompt 1 submitted only after the client observes prompt 0's first SSE token.

The last mode is an explicit client handoff, not proof of a particular GPU
graph or effective batch size. Submission order is not a server execution
order guarantee. No per-step server trace was collected in this replay.
Both arms complete, emit full 64-element token/timing vectors, acknowledge
drain and release their GPU processes. The 32 GiB MemAvailable reserve is
retained; an observed serving snapshot had 102,157,656 KiB available.

| Comparison | Result |
| --- | --- |
| Each solo prompt, repeat 0 versus repeat 1, within each arm | All four pairs equal |
| Prompt 0 alone, 2K versus 8K | Both repeats equal |
| Prompt 1 alone, 2K versus 8K | Both repeats differ at output index 2; 44/64 tokens differ |
| Prompt 0 paired immediately, 8K versus its 8K solo | Both orders/repeats differ at index 4; 41/64 tokens differ |
| Prompt 0 with first-token handoff, versus its arm's solo | Equal in both arms/repeats |
| Prompt 1 with first-token handoff, versus its arm's solo | Differs in both arms/repeats |
| Same mode repeated under 8K | All eight request-vector pairs equal |
| Same mode repeated under 2K | Immediate `pair01` changes on both prompts; other six pairs equal |

Indices are zero-based. A changed greedy token alters all later inputs, so
44 different tokens do not mean 44 independent numerical faults. Stable
repeat outputs do not independently prove absence of races or model quality.

The arms were measured sequentially, 2K then 8K, with no excluded warmup or
counterbalanced timing campaign. Raw times are retained for diagnosis but
are not substituted for the preceding ABBA performance campaign. No
vLLM/SGLang baseline is remeasured here.

## Cross-phase attention on .178

Host: Spark GB10, sm121, GPU UUID
`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
Compiler: CUDA 13.0.88, nvcc SHA-256
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The selected artifacts are copied from the .179 frozen routed bundle, without
regenerating kernels:

- Prefill: `lunaflux_attention_prefill_tile_compiler_v1`, c322,
  `strict-natural-exponential-v1`, query/KV tiles 64/64.
- Decode: `lunaflux_attention_decode_tile_compiler_v1_owned8_blockwise_f32_v4`,
  c468, `owned8-blockwise-f32-probability-v4`, KV tile 32.
- Partitioned decode: the same artifact's split partial/merge chain, all eight
  partitions and the merge launch included.

For history 1,024/32,512, rows 1/2, partitioned/nonpartitioned decode and two
repeats, prefill and decode receive identical query, positions, page mapping
and KV. Phase-specific count/CSR arguments are constructed separately.
Each row contains exactly one new query. The diagnostic is check-only:
cross-phase timings cannot be reported as optimization wins.

All 16 checks exit zero with empty stderr, unchanged KV and the existing
sampled FP64 softmax oracle bound. Every pair is non-bitwise: maximum output
difference is **0.00048828125**. Decode's sampled oracle errors range from
approximately **0.0002375 to 0.000242923**, below the unchanged 0.003 bound.
The probe reports zero local allocation for these kernels. These small synthetic errors
do not establish full-model greedy stability or identify a defective kernel.

The runner uses user-systemd MemoryMax 8 GiB, zero swap and a 900-second limit,
with an additional 32 GiB MemAvailable check before each invocation. It was
compiled natively on .179 and transferred to .178, which lacks Moon tooling.
No production dependency, runtime JIT or driver setting was introduced.

## Automation, failures and saved results

Added MoonBit automation:

- `benchmarks/gpu_pipeline/ako_batch_replay.mbtx`: exact saved-input replay and
  first-token handoff; bodies are loaded as runtime JSON, not embedded as AST.
- `ako_batch_replay_report.mbtx`: preserves complete vectors, checks 64-token
  completeness, and reports solo/repeat/cross-chunk first divergences.
- `ako_phase_parity.mbtx`: bounded cross-phase synthetic diagnostic, retaining
  numerical failures rather than relabeling them as passes.
- `selected_policy_probe.cu`: explicit check-only cross-phase mode. Existing
  production sources and numeric contracts are untouched.

The first .179 attempt failed before requests: embedding two 32K bodies as
Moon source exhausted the compiler stack. Its complete failure logs are
preserved under `campaign`; corrected results are under `campaign-r2`.

Remote archives and verified local SHA-256:

- `.179`: `/home/wlc004s/lunaflux-ako-batch-isolation-20261005.TdP1Cpeh.tar.gz`,
  `d9fdaafa0082c3bd3e9f6d52d99b0a10b52b004e86d45b0b207cb58ba06c42a8`.
- `.178`: `/home/wlc003s/lunaflux-ako-phase-parity-20261005.ExrMud5W.tar.gz`,
  `84222c92e14ffce3bea72239f2e28d0697db7daedbd0dcf08eaf839272196c68`.

Both were downloaded without overwrite into
`/tmp/lunaflux-phase-parity-transfer-20261005.WRiQ0kOq`, and local archive
hashes match. Build caches are excluded from the serving archive, but scripts,
failure logs, raw successful outputs, memory observations and audit are kept.
Focused native warning-denied checks pass for all three scripts; replay and
report regression tests pass. The diagnostic CUDA fixture compiles on sm121.

The next meaningful experiment is a disposable, selected-route/logit-margin
trace at the first divergence, starting with prompt 1 alone. A scheduler-only
fix cannot explain that solo cross-chunk failure, and making the tolerances
looser would not be a compiler optimization.
