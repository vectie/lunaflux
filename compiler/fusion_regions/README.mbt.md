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

Attention ingress is the first integration: its known full, separated-producer
and unfused implementations map to validated region partitions with an
observable positioned activation and an ordered KV write. The runtime bundle
exporter selects those partitions from whole-span observations. In the absence
of observations or explicit evaluation, it preserves unfused semantic stages;
that conservative policy is not a speed claim.
