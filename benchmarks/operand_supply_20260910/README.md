# Packed matrix transport measurements

This snapshot is the exact offline tuning input used with implementation
`ed6f572a6d6ca39bf6bb365c4e8b0e634b0a55e2` in the September 10 GPU comparison.
It is a device/shape/backend-specific benchmark record, not a universal
recommendation or a model-name branch in the compiler.

The head records replace the historical single-output-tile distributions:

| Bucket | Measured tokens / selected rows | Output tiles / workgroup | Median ns | Trials |
| --- | --- | --- | ---: | ---: |
| 8 | 8 / 8 | 2 | 740713 | 3 |
| 256 | 256 / 32 | 2 | 808126 | 3 |

These are CUDA-event, repeated immutable-CUBIN measurements, not client
latency or NCU replay times. The row-256 kernel was additionally checked at
2, 4, 8, 16, 17, 32, 33 and 65 tokens with valid selected-row counts. Both
simple and varied-exponent BF16 operands matched the previous implementation
bit-for-bit. Four output tiles had effectively the same row-8 latency, with
more shared storage; the smaller two-tile alternative was selected.

The other three records retain the historical measurements from
`../operation_optimization_20260908/projection-tuning.v1`; their timing fields
are **not** fresh measurements of this implementation. No measurements for
one bucket are relabeled as measurements for another bucket. The existing
artifact contract may select a wider compiled envelope for smaller demands.

The scope line pins the GPU UUID, CUDA backend/architecture and toolchain.
The pure strategy and ordered reduction are unchanged; the CUDA lowering
implements packed matrix loads, independent operand-copy domains and direct
unique-owner stores. No benchmark search runs in the serving request path.
