# Long-input three-engine retest and compiler implementation audit

## Scope and measurements

RTX 5060 Ti, Qwen3-0.6B BF16, identical repeated/truncated token-ID prompts,
greedy generation, forced output length, prefix reuse disabled. Each cell has
one warmup and five measured bursts. LunaFlux ran first, followed by vLLM
0.24.0 and SGLang 0.5.2 on the same day and device; no GPU workloads overlapped.
This is sequential, not randomized/counterbalanced, and not a sustained-load
or p99 study. Engine-native scheduler/kernel settings are retained: these are
end-to-end comparisons, not identically scheduled operation replays.

LunaFlux source snapshot is
`959032e9312508b7e2afc4bb78460ccfa866853ee05ee755488e621853cb7ef2`
(`82f00517` plus existing worktree changes). All standard/fused artifacts were
regenerated. The measured attention table selects c322; it does not select c324.
Runtime preparation's `gpu_tested=false` receipt predates testing; the later
`current/lunaflux/RESULT.txt` records the completed serving campaign.

Mean output tokens/s includes prefill:

| Input/output | C | LunaFlux | vLLM | SGLang |
|---|---:|---:|---:|---:|
| 512/64 | 1 | 211.65 | 244.29 | 238.46 |
| 512/64 | 8 | 1033.52 | 1180.81 | 1135.26 |
| 512/64 | 16 | 1375.68 | 1648.43 | 1454.50 |
| 1528/32 | 1 | 149.42 | 178.22 | 175.06 |
| 1528/32 | 8 | 367.29 | 443.99 | 418.03 |
| 1528/32 | 16 | 405.45 | 493.07 | 467.07 |
| 3072/32 | 1 | 107.54 | 127.20 | 122.72 |
| 3072/32 | 8 | 184.07 | 222.88 | 216.25 |
| 3072/32 | 16 | 195.03 | 232.35 | 228.94 |
| 4096/64 | 1 | 116.53 | 140.79 | 136.94 |
| 4096/64 | 8 | 214.69 | 255.36 | 251.85 |
| 4096/64 | 16 | 175.32 | 266.38 | 260.71 |

SGLang has retained outliers, not discarded samples: 512/C16 trial 5 is
877.46 tok/s versus approximately 1595–1600 in the other four trials;
4096/C16 trial 3 is 229.39 versus approximately 268.5 otherwise. Means include
them. Five bursts do not establish robust tail statistics or explain their cause.

Long-input client timing (LF / vLLM / SGLang):

| Input/output/C | Mean batch wall ms | Mean TTFT ms | Mean post-first-token ms/token |
|---|---|---|---|
| 3072/32/1 | 297.6 / 251.6 / 260.8 | 128.6 / 115.4 / 119.6 | 5.39 / 4.26 / 4.44 |
| 3072/32/8 | 1390.8 / 1148.6 / 1183.8 | 623.9 / 511.1 / 506.8 | 23.39 / 19.30 / 21.67 |
| 3072/32/16 | 2625.2 / 2203.6 / 2236.4 | 1144.0 / 966.6 / 930.6 | 44.65 / 36.47 / 41.89 |
| 4096/64/1 | 549.2 / 454.6 / 467.4 | 178.6 / 160.6 / 165.6 | 5.84 / 4.60 / 4.73 |
| 4096/64/8 | 2384.8 / 2005.0 / 2033.0 | 888.7 / 737.3 / 703.4 | 22.98 / 19.19 / 21.00 |
| 4096/64/16 | 5840.8 / 3844.2 / 3943.4 | 1644.6 / 1422.6 / 1310.4 | 62.47 / 35.79 / 39.58 |

4096/C16 LF wall time is 1.519x vLLM and 1.481x SGLang. Its large gap is
not exclusively before the first token. The client token intervals include
remaining prefill interference, scheduling and transport; they do not isolate
steady-state decode GPU cost. New selected-kernel traces at this coordinate
are required before assigning the gap to a specific kernel or graph fallback.

All 500 measured requests per engine completed with the expected token counts.
At 3072/C8, all three engines produced two sequences; LF also produced two at
C16 while each baseline produced one. Baseline 3072 sequences belong to LF's
observed sequence pool, not necessarily the same ordinal request's sequence.
At 1528/C1 SGLang's five sequences differ from LF's. Other cells have matching
sequence pools. This is not blanket deterministic equivalence or quality validation.

## Which proposed optimizations are actually unfinished?

The earlier five mechanisms are present; their integration/coverage is uneven.
This audit reads current source and the tested runtime, not only old TODO prose.

| Proposed work | Present now | Concrete remaining work |
|---|---|---|
| Operand segmentation / partial evaluation | Gate/up phased transfer uses pure disjoint vector-owner segments and invariant source addresses | Generalize that treatment to all projection paths. QKV matrix transfer still renders Q/K/V base selection inside `stage_operands`; inspect selected-code instruction/latency before changing it |
| QK → softmax → PV ownership | Query-owned register fold, exact masked-exp if-conversion, BF16 pair packing, shared address products, history specialization; c322 is in this runtime | Remaining move/address/control and resource costs are not zero. The exact numeric micro-optimizations live in CUDA lowering, not a universal optimization pass over arbitrary IR |
| Query tails / tiles / waves | Tail guards, resource/wave cost interfaces, Q64/K32 c324, split partial/merge variants, graph bucket routing exist | Per-query/history/batch winning schedules are not deployed as a measured dispatch table. Current single/wide prefill alias one selected frontier variant; larger Q128/K128 experiments lost and were not integrated |
| Dense-current / paged-history read views | Shared logical stream, metadata-v1, preserved persistent KV writes, c322 history-only specialization are wired | Full QKV fusion is not active in this runtime: independent QKV plus fused QKNorm/RoPE/KV-write remains. Cross-op selection should compare full/partial/unfused chain cost, not assume maximal fusion wins |
| Resource feedback / autotuning | Backend-neutral resource budget/observations and offline latency records exist; actual exported 318/322/324 measurements select c322 | Qwen candidate exporter does not call `with_resource_feedback`; constructors default to no budget/empty register observations. Wire real post-lowering resource facts and complete per-bucket measurements into selection |

Related items must not be called absent:

- GEMM already has two transfer slots and bounded fragment lookahead. However,
  `ProjectionFoldPipeline::stage_count` returns 2, fragment count uses a
  geometry heuristic, and transfer-window limits are rules, not a complete
  measured multi-stage search. Three/four/six-stage profitable scheduling is
  not established by the present implementation.
- Independent prefill/decode kernels, split-K partial/merge, continuous batching,
  and prefill/decode/mixed execution-graph buckets exist. The remaining task is
  actual 4096/C16 row/token/padding/owner/replay/metadata/idle-gap attribution,
  including shrinking batches, rather than implementing batching from scratch.
- Residual+RMSNorm and partial attention ingress are in the tested bundle.
  Full ingress source/export support exists but is not selected here.
- Current output variation remains unresolved. Any scheduling or fold change
  needs controlled batch-membership/logit comparison, not only token counts.

Source anchors: `kernels/luna_projection_tile_compiler/fold_pipeline.mbt`,
`kernels/luna_cuda_projection_aot/source_matrix_pipeline.mbt`,
`kernels/luna_cuda_projection_aot/source_sibling_phased_transfer.mbt`,
`kernels/luna_cuda_attention_tile_source/source_query_owned.mbt`,
`kernels/luna_cuda_attention_tile_source/source_query_stage.mbt`,
`kernels/luna_cuda_attention_tile_aot/{construct,resource_feedback}.mbt`,
`cmd/lunaflux_qwen3_bf16_candidate_export/main.mbt` (frontier creation/tuning
and `qwen_wide_query_prefill_variant`), `engine/device_step/graph_bucket.mbt`.

Next order: profile actual 4096/C16 versus C8 and 3072/C16; identify selected
per-step work and resources; then implement per-bucket selection/resource
feedback and the demonstrated projection/fusion gaps. A half-vLLM target is
still unproven. No kernel or production deployment was changed in this audit.

## Reproduction

LF: `/run/lunaflux-toolchain-4896771-20260913/attention-current.vX2pXE/current`.
Baselines and `THREEWAY.json`: `/tmp/lfe2e.2sP5zM`.
Identical `client.mbtx` workload, one engine at a time, runners stop their own
process groups and confirm GPU idle. The first vLLM attempt is preserved in
the LF campaign's `current/vllm`: its TMPDIR made a ZMQ IPC socket path exceed
107 characters, before inference. Moving the retry to the short root fixed
the harness issue without changing model/engine parameters.

Downloaded baseline archive:
`/tmp/lunaflux-instruction-fix.9n6ITe/threeway-baselines-20260914.tar.gz`,
SHA-256 `d2bbb24b6f92cc9783261fd11bd6cc0c45b91a4e1c8d2c7743ef5afa249cb0f4`.
LF archive: `/tmp/lunaflux-instruction-fix.9n6ITe/e2e-results-20260914.tar.gz`,
SHA-256 `f98d0109e44636a40048f35bfb77acdfbb104d90fd19855199878bcc1cff9c30`.
Both local hashes match their remote originals. The three-way summary script
passed a warning-denied build; no production source was modified.
