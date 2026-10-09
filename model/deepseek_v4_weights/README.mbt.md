# DeepSeek V4 weight-storage boundary

Official V4 checkpoints use mixed storage. Main quantized matrices use dynamic
finite E4M3 with 128×128 block scaling and UE8M0 scales. Instruct checkpoints
add packed E2M1 FP4 experts with 32-element scaling, while base checkpoints
retain FP8 experts. BF16, F32, and I64 auxiliary tensors are also present.
The first three decoder layers carry the official row-major I64 checkpoint
`ffn.gate.tid2eid` table with shape `[vocabulary_size, experts_per_token]`.
Startup schema evidence also records the official output-A conversion from
E4M3 plus its 128×128 UE8M0 scale grid to BF16 group-major
`[groups,output_rank,input_per_group]`. It does not perform that conversion.

The family-wide binder owns a metadata-only safetensors header view and an exact
semantic vocabulary. Its binder validates canonical names, dtype tags, shapes,
shard-local byte ranges, global uniqueness, and the official tensor cardinality
for all five profiles. Roles cover hybrid attention and its indexer/compressor,
hash and score routing, mHC, shared/routed experts, MTP, DSpark, quantization
scales, and the fixed E2M1 codebook.

Successful metadata binding joins authenticated content with the canonical
advanced execution-plan digest. The family-wide host/device materializer still
does not execute a full checkpoint.

A separate compact expert adapter now maps w1/w3/w2 checkpoint names to shared
precision representations: FP4 uses row/block32 E2M1+UE8M0; FP8 uses E4M3 with
128x128 UE8M0 grids. integration/deepseek_v4_packed_expert binds and streams
these planes into the generic expert bank and supplies dynamic block128 FP8
activation semantics. This executable expert path does not imply a complete
decoder/DSpark runtime.
