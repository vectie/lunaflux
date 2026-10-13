# Contiguous prefill physical plan

Maps one logical prompt to exact contiguous row tiles within a declared AOT
capacity. No padding becomes semantic history and only the final logical row
samples. Recurrent state still progresses in sequence order inside each tile;
this is not permission to parallelize a recurrence or expose future sparse keys.

The startup prompt adapter serializes this immutable plan to existing worker
frames. The same row capacity must reach AOT export, allocation and execution.
