# Ordered fusion regions

This pure offline/startup pass partitions arbitrary SSA DAG dependencies in a
declared topological order. It does not reorder arithmetic or search
noncontiguous node sets. Graph owners describe observable outputs and ordered
effects; backend catalogs supply implementations of contiguous regions.

Every region must export values observed by the caller or consumed outside the
region. A boundary effect forces a singleton; ordinary ordered effects remain
in original node order. An exact cover executes every node once, including
effects. The planner cannot invent an executable fused implementation.

`select_regions` uses dynamic programming for a global minimum-cost serial
cover rather than choosing the largest fusion greedily. Region costs include
all materialization, launch and synchronization work. Scratch ceilings apply
to each region's reusable workspace and local storage; persistent activations
remain owned by the execution memory planner. Stable catalog IDs break ties
lexicographically, independent of enumeration order.

`select_partition` separately compares whole-chain measurements. It never
replaces end-to-end measurements with sums of isolated microbenchmarks.
Measured and estimated records cannot be mixed, and timing records require at
least three samples. The caller owns measurement identity and comparability.

It now uses `PartitionFrontier`: retain latency/workspace/local-storage
tradeoffs before resolving the resource envelope. Pruning is allowed only
inside one caller-owned continuation class (complete live-value layout,
numerical policy, effects, ownership handoff and cost-relevant continuation
context). Layout equality alone is insufficient when cache/residency or launch
composition differs: keep separate classes or measure the complete chain.
Different classes remain distinct even if one has lower isolated latency and
storage. `Materialized`
preserves the existing materialized-boundary contract. Class IDs are scoped to
one comparison, not global ABI identities; the planner does not invent a
conversion between classes.

`PartitionFrontier::select` resolves a class and resource budget without
mutating the frontier. Stable IDs break latency ties even when storage differs,
preserving existing selection behavior. Persistent activation memory remains
outside the reusable-scratch ceiling. The existing ingress exporter calls this
path through `select_partition`; it does not yet supply alternative consumer
layouts. See the [execution-economics policy](../../docs/EXECUTION_ECONOMICS_2026-10-08.md)
for the architecture and remaining integrations.

`ConnectedPlan` composes supplied executable regions before pruning. Adjacent
implementations must agree on the complete live-boundary contract, and their
regions must form a valid ordered cover. `select_connected_plan` accepts only
whole-chain measured costs, reuses `PartitionFrontier`, and returns an immutable
vector of implementation IDs. An exporter must bind that vector together, not
independently reselect its components. Serial scratch is a maximum; overlapping
execution and persistent storage still require their respective planners.
Unknown local-memory usage is represented as `None`, never as zero. A finite
local-memory constraint excludes such a plan. With no such constraint, that
dimension is projected out for all alternatives; external artifact admission
must still establish hardware legality.

The Qwen bundle export command has an opt-in offline integration for four
existing attention/projection execution plans. It binds both attention slots,
projection policy and the chosen artifact's decode-route table. These plans
share a materialized BF16 boundary: this is not yet a catalog of alternative
nonmaterialized consumer layouts, nor evidence of an architectural speedup.
See [connected selection](../../docs/COMPILER_CONNECTED_SELECTION_2026-10-09.md).

```mbt check
///|
test "defer selection until the resource budget is known" {
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=true, effect=Pure),
  ])
  let partition = @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=1, [0]),
  ])
  let choices = @fusion_regions.PartitionFrontier::new([
    @fusion_regions.PartitionAlternative::new(
      id=1,
      partition,
      @fusion_regions.Cost::estimated(nanoseconds=10L),
      workspace_bytes=64L,
      local_bytes=0L,
    ),
    @fusion_regions.PartitionAlternative::new(
      id=2,
      partition,
      @fusion_regions.Cost::estimated(nanoseconds=20L),
      workspace_bytes=0L,
      local_bytes=0L,
    ),
  ])
  inspect(choices.alternatives().length(), content="2")
  inspect(
    choices
    .select(
      continuation_class=Materialized,
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
    .implementation_id(),
    content="2",
  )
}
```

Attention ingress is the first integration: its known full, separated-producer
and unfused implementations map to validated region partitions with an
observable positioned activation and an ordered KV write. The runtime bundle
exporter selects those partitions from whole-span observations. In the absence
of observations or explicit evaluation, it preserves unfused semantic stages;
that conservative policy is not a speed claim.
