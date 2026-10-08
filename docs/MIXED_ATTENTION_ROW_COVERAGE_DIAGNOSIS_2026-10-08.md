# Unified mixed attention: missing tail writes, not proven rounding noise

## Finding

The previous **diagnostic worker rewrite is incorrect** for the partitioned
prefill tail. This is not a newly demonstrated defect in the unchanged
production worker: its separate decode companion supplies the missing rows.

The diagnostic expanded query-tile metadata to all rows and then removed
decode companions unconditionally. The ordinary matrix kernel writes all rows
described by that metadata. The partitioned partial also consumes metadata,
but its final merge independently enumerates `candidate_row < prefill_count`.
Expanding the partial's metadata does not expand the merge's output domain.

Relevant source:

- `engine/device_step/attention_metadata.mbt`: original prefill-only metadata.
- `kernels/luna_cuda_attention_tile_source/source_query_metadata.mbt`: matrix
  partial consumes explicit tile records.
- `kernels/luna_cuda_attention_tile_source/partitioned_source.mbt`: prefill
  merge enumerates only prefill rows; decode merge owns decode rows.
- `engine/device_step/attention_phase_executor.mbt`: original startup rewrite
  retains a decode companion after the final attention writer.
- `benchmarks/gpu_pipeline/mixed_unified_serving.mbtx`: the prior diagnostic
  mechanical rewrite discarded those companions across all route families.

```text
Correct partitioned mixed graph:
  prefill partial → prefill merge → decode partial → decode merge → projection
                     writes P                         writes D

Incorrect diagnostic tail:
  all-row partial → prefill-only merge → projection reads P + stale D
```

## Direct physical reproduction

The regression reuses the exact serving bundle's selected p8 module, not a
newly compiled replacement kernel. The workload is the trace's final mixed
vector: `106:8086,1:8215,1:8211,1:8207,1:8203,1:8199,1:8195,1:8220`.
Queries/keys are zero and each request's V is its row index plus one. Every
output element therefore has an independent exact constant-row oracle.
Physical pages are permuted. Before each of eight repeats, every output is
poisoned with alternating +1024/-1024; valid outputs cannot equal either.

| Executed chain | Unwritten output elements per invocation | Incorrect prefill values | Numerical result |
| --- | ---: | ---: | --- |
| Original partitioned tail + decode companion | 0 | 0 | Every output exactly matches oracle |
| Ordinary matrix kernel, all-row metadata | 0 | 0 | Every output exactly matches oracle |
| Partitioned tail, all-row metadata, no companion | **14,336** | 0 | All seven decode rows retain poison |
| Same tail with decode companion retained | 0 | 0 | Every output exactly matches oracle |

There are 231,424 BF16 output values per invocation. The 14,336 missed values
are precisely `7 decode rows × 16 heads × 128 dimensions`. Across eight
repeats, all 114,688 missing writes are reproduced, with unchanged KV buffers.
The corrective companion here is a **probe control**, not a deployment.

The first reproduction used the probe's narrow tail grid. The final-source
recheck uses the actual captured grids: wide `63×16×1`, tail partial
`64×16×8` plus merge `64×16×1`, decode partial `8×8×8` plus merge `8×16×1`.
The source guard permits reusing the probe's ABI loader without another main
function; production binaries and kernels are not changed.

## Saved serving results corroborate the cause

In the earlier ABBA run, all 40 within-arm comparisons retained the first
token for both control and unified arms. The unified sequences later differed
in 36/40 comparisons. For **all 36 differences**, the common-prefix length
equals the earlier affected tail-token index from the two compared waves.
Different concurrent admission order moves that index, making a deterministic
missing write appear to be output nondeterminism.

The tail index is inferred from the existing per-request timestamps: the
decode token issued with the final request's first token. Timestamp matching
is bounded to 20 ms; this is correlation with the physical reproduction, not
a replacement for a captured per-request execution ID. The remaining 4/40
unified comparisons are identical. Control has 5/40 differences, none starting
at that predicted tail index; their cause is not established by this audit.

## Why prior gates did not catch it

1. **Replay selection was incomplete.** All 29 vectors used the ordinary wide
   matrix artifact. The final serving tail actually uses a different
   partitioned partial/merge pair. Exact row lengths alone did not make the
   replay an exact selected-route test.
2. **Dispatch proof was insufficient.** The trace confirmed companions were
   removed, but that absence was mistakenly treated as successful propagation
   without checking the remaining writers' row coverage.
3. **Sanitizers do not prove output completeness.** Output storage is a valid,
   already-initialized activation arena. Leaving it unchanged is neither an
   illegal address nor necessarily a race or uninitialized read. The poisoned
   output/full-element oracle detects what those tools cannot.

The earlier 2.63% serving throughput increase is withdrawn as a valid
optimization result. It is a timing observation from an incorrect worker.
The unchanged control's 104.13 tok/s remains valid for its measured workload.
The ordinary-kernel synthetic chain measurements remain limited replay data;
they do not validate this serving substitution or whole-model quality.

## Architectural repair direction

No extra model-specific branch or runtime validation pass is needed. Keep the
functional compiler boundaries and strengthen the existing physical/effect
plan: each composed chain must retain its **read domain, partial-state domain,
and final write domain**. A startup substitution may remove a companion only
when final writers cover all live rows before the downstream consumer, without
unintended overlap. Metadata coverage alone is not that proof.

Two legal implementations remain to compare after correctness:

- Retain the existing decode companion for prefill-only partitioned merges.
- Explicitly lower an all-row merge together with the all-row partial, binding
  the new numeric/write-domain contract and validating the complete chain.

Regression coverage must include ordinary/partitioned routes, every tail
bucket, mixed row ordering, poisoned activation storage, and repeated fixed
inputs. Then rerun serving and teacher-forced/logit checks before treating any
speed change as a win. This investigation does not implement or promote that
production repair, establish model-quality parity, or explain the original
remaining vLLM/SGLang performance gap.

## Reproduction identities

- Spark .179, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI
  `0000000F:01:00.0`.
- CUDA 13.0 nvcc SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Frozen serving bundle:
  `f7afc9b5a0a0a6fa148ca027c26c3c4c9f14b6580550d743c443d3ade94fb58a`.
- Extracted selected p8 module:
  `1c7e131ecd4fddc6eeac166ee75d13f9acfbca725acee7d901afee64d136e654`.
- Evidence root:
  `/home/wlc004s/lunaflux-mixed-numeric-20261008.es1leHBc`.
- Helpers: `audit_mixed_row_coverage.cu`, `run_mixed_numeric_audit.mbtx`,
  `analyze_mixed_divergence.mbtx` under `benchmarks/gpu_pipeline/`.

Only diagnostic probes/reporting changed locally. GPU work is serialized under
an 8 GiB/no-swap user-systemd limit and a 32 GiB MemAvailable admission floor.
The initial runner parse failure is retained; it performed no GPU work.

## Completed validation

The exact-grid final-source rerun reproduced the same missing-write count in
all eight repeats. Compilation had empty stdout/stderr. Memcheck, initcheck,
racecheck, and synccheck all reported zero errors/hazards, including while the
explicit output oracle detected the bad chain. The three MoonBit diagnostic
helpers each passed their focused native test. No new serving performance
result was collected, and the GPU was idle after the audit.

The sealed archive was downloaded without overwrite to
`/tmp/lunaflux-mixed-row-audit-verified-20261008.Kwa2im1l/` and extracted into
its `evidence/` subdirectory. The local outer SHA-256 matches the remote seal:
`ca9dc007d4bb4754f75f1382c9697f758fc1f60ed298551af12bbd854203749c`.
Every entry in the extracted `FILES.sha256` verified. The archive includes
the original failed setup, both successful probes, all sanitizer output,
selected module/source/recipe, user-unit journals, and the saved-sequence
divergence analysis. This download note is outside that immutable archive.
