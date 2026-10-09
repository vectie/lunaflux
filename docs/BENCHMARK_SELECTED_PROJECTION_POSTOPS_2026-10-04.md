# Selected projection/post-op follow-up

## Result and scope

The current measured excess is in **executed projection/post-op chains**, not
an unverified assumption about attention or bank conflicts. The full-step
ledger records 2,700.035 ms of exclusive LunaFlux projection/post-op activity,
versus 2,215.684 ms for vLLM and 2,334.602 ms for SGLang: excesses of
484.351 ms and 365.433 ms respectively. These are profiled activity totals,
not ordinary throughput measurements.

This follow-up preserves the exact query/history vectors from the existing
capture. It does not assign generic library GEMM/GEMV operations from their
launch grids. SGLang has no exact common full-step vector in this capture, so
the operation tables below compare only the exact LunaFlux/vLLM work matches.

## Current selected paths, not historical candidates

For one 28-layer forward step:

| Executed LunaFlux operation | `2048:2048`, ms | `1:4097,2047:2047`, ms | Selected configuration |
| --- | ---: | ---: | --- |
| Gate/up including gated activation | 16.048 | 16.142 | 1,536 CTAs, 512 threads, 64 registers/thread, 40,960 shared bytes |
| Full QKV/QKNorm/RoPE/KV-write | 11.953 | 12.023 | 32 × 32 CTAs, 128 threads, 116 registers/thread, 32,768 shared bytes |
| MLP down | 6.897 | 6.553 | 256 CTAs, 128 threads, 218 registers/thread, 49,152 shared bytes |
| Attention output projection | 4.914 | 4.854 | 512 CTAs, 256 threads, 80 registers/thread, 40,960 shared bytes |
| Projection/post-op whole chain | 39.816 | 39.575 | Includes the shared-chain helper operations |
| vLLM projection/post-op whole chain | 33.477 | 31.726 | Same complete query/history vectors |
| Measured whole-chain excess | 6.339 | 7.849 | Kernel activity, not completion time |

The selected full ingress is now **64 token rows per head**, with one column
window, cached rotary preparation, and asynchronous operand staging. The old
16-row/multiple-column-window diagnosis is not the current executed path.
Its headwise projection geometry may still restrict reuse, but that needs a
new exact-chain experiment; the old diagnosis cannot establish the present
cost by itself.

The selected long gate/up is still the ordinary split-sibling path, not the
previous co-owned or retained alternatives. Those alternatives already lost
whole-chain A/B tests on this hardware and must not be reintroduced merely
because they execute fewer instructions.

## Source-level gap to measure next

The exact serving source for long gate/up uses a 64 × 64 output tile and K32
operand chunks. Its segmented producer precomputes source pointers, loads
128-bit global vectors into registers, then stores those registers to shared
memory. It issues the next transfer **after** consuming the current tile.

The reason is explicit in the generic physical planner:
`ProjectionPhysicalPipeline::new` replaces the supplied transfer execution
with `SerialTransfer` whenever `SegmentedSiblingTransport` is selected.
The terminal producer accordingly emits synchronous
`ld.global.v4.u32 → st.shared.v4.u32`, rather than an asynchronous transfer.
This is a real executable restriction, not a missing pass count or a model
branch. However, the existence of this restriction does **not** establish
that changing it is faster.

The next bounded experiment should retain the same arithmetic order, tile
geometry, segment/vector ownership, layout, and epilogue, while allowing the
typed physical effect plan to express an asynchronous segmented producer.
The compiler should own issue/consume/await/publication ordering; CUDA lowering
should only realize those effects. Compare:

1. Exact selected gate/up and down hardware counters, including long and
   rows16 launch configurations.
2. Unprofiled alternating whole-MLP-chain A/B, with component timings for
   gate/up and down. A faster component with a slower chain is not a win.
3. Bitwise complete workspace/output equality, masked-row correctness, and
   memcheck/racecheck/synccheck.
4. Selected-source and cubin hashes plus an executed-path trace after any
   positive compiler change is propagated into a serving bundle.

No such asynchronous segmented change is enabled by this report, and no new
physical improvement is claimed. GPU work awaits the shared-host serialized
slot; no GPU workload was launched for this offline follow-up.

## Remaining attribution boundary

For `2048:2048`, vLLM executes an unresolved generic GEMV taking 1.956 ms,
followed by generic reduction/copy work. For the matched tail `1:4350`, the
reference has 7.408 ms of unresolved operation activity. Therefore a table
showing zero reference vocabulary-head/projection time would be misleading.

The trace contains forward-level work markers and CCCL scopes, but not
projection-specific operator names or GEMV arguments. The new offline
summarizer deliberately retains these operations as unresolved. To resolve
them, a reference capture needs semantic operation markers or authoritative
captured call arguments; launch geometry alone is insufficient.

## Reproduction and retained output

Tool: `benchmarks/gpu_pipeline/summarize_projection_selected_subops.mbtx`.

```sh
moon run benchmarks/gpu_pipeline/summarize_projection_selected_subops.mbtx --self-test
moon run benchmarks/gpu_pipeline/summarize_projection_selected_subops.mbtx \
  EXISTING_CAMPAIGN_ROOT NEW_OUTPUT_JSON \
  '2048:2048' '1:4097,2047:2047' '1:4350'
```

The tool is read-only on the captures, uses exact work vectors, and refuses
to overwrite its output. The classification tests explicitly retain generic
GEMV/reduction as unresolved and avoid confusing FlashAttention's embedded
`cutlass::bfloat16_t` type with a projection GEMM.

- Remote output:
  `/home/wlc004s/lunaflux-selected-projection-20261004.mm5YvPnK/selected-suboperations-v2.json`.
- Local downloaded output:
  `benchmarks/results/selected-projection-20261004.4MqHcP85/selected-suboperations-v2.json`.
- Matching local/remote SHA-256:
  `b0eaf06e4cdebadbebc517ff3d1579d2b7630e5afa4dda0b64a6f62e51bf3885`.
- Inputs are the preserved
  `/home/wlc004s/lunaflux-full-step-measure-v2-20261004.tIVT3hCB/campaign/*/work-shapes.json`
  files; original captures and serving artifacts were not changed.

See [full-step attribution](BENCHMARK_FULL_STEP_ATTRIBUTION_2026-10-04.md)
and [previous sibling ownership/transport experiment](BENCHMARK_GATE_OWNERSHIP_TRANSPORT_2026-09-30.md)
for the wider ledger and prior losing alternatives.
