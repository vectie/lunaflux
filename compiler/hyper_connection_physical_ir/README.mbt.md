# Ordered hyper-connection physical plan

This pure plan distributes independent control outputs without changing the
precision IR's ordered products, folds, nonlinear expressions or publications.
Cooperative product tiles are published before their ordered consumer and cannot
be reused until that consumer retires. Separate launches publish square inverse,
control dots and collapse. Device lowering owns worker mapping and synchronization.

The frame retains one inverse and `streams * (streams + 2)` F32 mixes per row.
All storage is counted at preparation and deterministically released. Comparison,
benchmark data, model names and CUDA instructions are absent from this package.

`PostSquareFusion` proves compatible adjacent residual publication and square
reduction, including the intermediate BF16 round. It is an explicit alternative,
not a performance guarantee. `ControlTailFusion` independently describes
control/collapse/normalization plus Sinkhorn matrix ownership. When the supplied
subgroup capability can hold all matrix cells, independent cells share that
subgroup while preserving each row/column sum's arithmetic order. Larger matrices
retain workgroup-local storage. Neither plan chooses itself from a model name.

The generic parallel IR rewrites adjacent launch coordinates and rejects any
effect/fork boundary inside a removed pair. Bound frames validate producer and
consumer storage before a fused queue is created; all resource ownership stays
with those frames until queue retirement. Offline probes and full-model
selection evidence live in `docs/GLM_HC_AND_VOCABULARY_TP_2026-10-18.md`.
