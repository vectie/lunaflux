# Window and compressed shared-KV attention

Consumes one pure window/compressed read-set plan. Learned IDs name compressed
rows plus explicit offsets; all-causal plans need no learned selection. Window
and compressed scores share a single sink-softmax denominator. Current chunk
K/V is read separately from the prior ring, so long chunks cannot destroy
history needed by early query rows. Copy/publication source is shared with the
existing window-only renderer, not a second retention algorithm.

Four effects reserve, attend, retain and publish. Reserve validates upstream
cache publication and compressed selection metadata before the transaction.
The initial correctness renderer uses shared scores and explicit BF16 output;
its static-shared resource limit belongs to CUDA lowering, not the semantic IR.
It is not a tuned throughput kernel or a whole-model numerical claim.
