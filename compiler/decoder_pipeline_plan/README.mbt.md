# Contiguous decoder placement

This pure startup planner partitions ordered resident layer footprints under
per-host memory ceilings. First/last text weights are counted only on their
owning stage. Every layer occurs exactly once. Suffix feasibility prevents a
greedy placement from stranding remaining layers on a smaller host.

Costs include actual weights, persistent state and layer workspaces. The caller
adds stage residuals, request metadata and its explicit memory reserve. This is
memory placement, not tensor parallelism, a throughput optimizer or offload.
Device transport and execution effects belong below the plan.
