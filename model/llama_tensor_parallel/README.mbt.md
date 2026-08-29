# Dense-Llama tensor-parallel rank planning

This package is the model-family planning boundary for ADR-0010. It derives one
immutable local-rank plan from an already validated dense-Llama `ModelPlan`,
its admitted BF16 weight bindings, and an exact `(rank, world_size)` pair. It
does not import a scheduler, device backend, collective library, filesystem, or
request contract.

The v1 placement policy is exact and fail-closed:

- embedding and language-model-head matrices shard safetensors rows (the
  vocabulary axis);
- Q/K/V and gate/up matrices shard rows (output channels);
- attention output and MLP down matrices shard columns (input channels);
- RMSNorm vectors are intentionally replicated;
- vocabulary, hidden, intermediate, attention-head, and KV-head dimensions
  must all divide the world size.

Every placement reports full and local extents plus its aligned final rank-arena
region. Its transfer recipe uses payload-relative source ranges. Replicated and
row-sharded tensors require one contiguous transfer. A column-sharded tensor
uses a scalar strided recipe with one segment per source row, so a loader can
copy each range directly to its final compact destination. The plan stores only
one fixed recipe per tensor; no rank materializes a complete copy of any tensor
declared sharded and no temporary shard buffer is required.

Collective sites are explicit and ordered by consecutive stable sequence IDs:
embedding sum-all-reduce, each layer's output-projection sum-all-reduce, each
layer's gated-MLP sum-all-reduce, then vocabulary/logits all-gather. World size
one produces the deterministic oracle placement with no collective sites.

This package does not create a rank group, allocate memory, read a file, launch
a kernel, or confer collective authority. A later private execution bridge must
authenticate group generation, schedule-plan sequence, collective sequence,
operation, rank, and world size before dispatch. A later sharded loader must add
the independently admitted safetensors payload base with checked arithmetic and
stream each recipe segment directly into the final local arena.
