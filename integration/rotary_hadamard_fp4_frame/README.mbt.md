# Prepared precision transform

This frame owns two bounded BF16 tensors and three functions; counts, positions
and input are borrowed from the containing graph. All three effects join its
existing queue. The transformed output feeds generic retained-row storage or a
query/index consumer without a diagnostic host copy or separate completion.

The DeepSeek adapter owns the choice between ordinary attention prefix E4M3
simulation and indexer full-vector Hadamard/E2M1 simulation. No model identity
or checkpoint tensor names enter this generic frame.
