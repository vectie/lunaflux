# Full-softmax top-K source

BF16 full expert logits feed FP32 max-shifted softmax, descending top K,
selected-probability renormalization and BF16 output weights. Lower IDs win equal
probabilities. The initial implementation orders both sums and performs routing
on thread zero; optimize only after measured numerical and timing evidence.

The ABI is `counts, logits, indices, weights`, where `counts[3]` is active rows.
Logits are packed `[row, expert]`; outputs are `[row, selected_rank]`, with int32
indices and BF16 values. The plan selects BF16 storage or exact widening to F32
storage for existing expert kernels. The BF16 value round occurs in both cases.
Router projection is a separate BF16 matrix operator.
Generated source is uncompiled and unqualified until target validation.
