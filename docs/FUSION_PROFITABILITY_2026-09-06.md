# Complete-span fusion selection

Semantic fusion legality and profitability are separate decisions. An immutable
backend-neutral comparison ranks complete, numerically equivalent schedules:
unfused, producer-separated (projection plus fused postprocessing), and fully
fused. Measurements must cover the same input/output boundary, including all
persistent KV effects and the attention consumer. A fused-kernel-only duration
is not comparable with an unfused complete-span duration.

`luna_fusion_strategy` accepts either complete-span measured medians or a fully
specified estimate (compute, memory traffic, synchronization, launch overhead).
It rejects mixed measured/estimated comparisons, duplicate alternatives and
insufficient samples. Resource limits filter alternatives before ranking.
Equal costs prefer the less fused schedule; input order has no effect.
Estimates are planning inputs, not benchmark results. Hardware-specific cost
calibration belongs to backend lowering, not this pure selector.

The Qwen fused-runtime exporter consumes measured comparisons through an
optional `--ingress-comparison` trailer after its original arguments:

```
--ingress-comparison FULL_NS PARTIAL_NS UNFUSED_NS SAMPLES
  PARTIAL_GRID_X PARTIAL_GRID_Y PARTIAL_GRID_Z PARTIAL_BLOCK_X
  PARTIAL_SHARED_BYTES PARTIAL_CUBIN
```

The caller must have qualified all three alternatives on the same device,
numeric contract, shape/profile and workload. These explicit CLI measurements
are not an automatically discovered or device-authenticated tuning database.
The existing ingress arguments continue to describe the full variant. Partial
selection retains the ordinary projection and binds the supplied postprocess
module. Unfused selection emits a residual-only runtime bundle: it must not
leave read-only attention installed without its KV-write producer. No trailer
preserves the original full-ingress behavior. No runtime file access, timing,
allocation or model-name cost heuristic is added.

This is an offline whole-profile choice. Per-bucket ingress alternatives and
their captured-graph memory budget are a separate remaining extension. A faster
isolated span is not an end-to-end performance claim.
