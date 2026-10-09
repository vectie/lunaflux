# DeepSeek V4 key/value parameter plan

This integration package makes one inert join between the family-owned
DeepSeek V4 checkpoint layout and the family-neutral key/value projection AOT
candidate. It authenticates the model and requirement identities, the
replicated `[512, hidden]` E4M3 code tensor, the `[4, hidden / 128]` UE8M0 scale
tensor, and candidate operands 2 and 3.

The plan does not interpret raw bytes, convert or materialize tensors, upload
device memory, compile CUDA, bind a launch, or authorize execution. Each of
those authority requests fails explicitly.
