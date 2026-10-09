# Pure two-node plan compilation

`compile` joins an existing semantic model, two family-produced BF16 rank plans,
an explicit network topology, and two memory budgets. It checks rank ordering,
model identity, numeric storage, complementary shard extents, canonical transfer
recipes into the same source tensors, matching operations and collective sites.
It neither discovers machines nor lowers model-family operations itself.

Weights use each rank's aligned arena size. System headroom is outside that
rank's runtime ceiling; activations/workspace, collectives and KV are separate
reservations. The canonical `kv/device_layout` planner derives aligned local
K/V segments. Common page capacity is the minimum of the two rank capacities
and the requested page ceiling. Memory cannot spill into the other node.
Workspace and collective reservations remain declarations until exact artifact
and runtime admission confirms them; this compiler output is not execution
authority.

Compilation computes one canonical SHA-256 identity over the model, ordered
node/NIC/device assignments, targets, physical/reserved memory, aligned KV
layout, rank tensors and transfer recipes, operations and collective sites.
It caches this immutable identity. No digest work belongs in the token step.
No arbitrary plan-deserialization or public digest constructor grants a plan.

```mbt check
///|
test {
  let memory = @two_node_plan.RankMemoryBudget::new(
    system_reserved_bytes=8000000000L,
    runtime_ceiling_bytes=120000000000L,
    activation_workspace_bytes=4000000000L,
    collective_bytes=1000000000L,
    kv_bytes=16000000000L,
  )
  assert_eq(memory.kv_bytes(), 16000000000L)
}
```

The black-box integration tests feed actual `model/llama_tensor_parallel`
outputs into this compiler; they also cover source/collective substitution,
per-rank exhaustion, aligned-page rounding, arithmetic overflow, immutable
input snapshots, and digest sensitivity. They are host tests, not GPU evidence.
