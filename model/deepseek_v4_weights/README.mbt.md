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

The existing generic safetensors metadata package is BF16-only, and LunaFlux's
numeric schema does not yet represent packed FP4 experts. This package therefore
owns a family-specific, metadata-only safetensors header view and an exact
semantic vocabulary. Its binder validates canonical names, dtype tags, shapes,
shard-local byte ranges, global uniqueness, and the official tensor cardinality
for all five profiles. Roles cover hybrid attention and its indexer/compressor,
hash and score routing, mHC, shared/routed experts, MTP, DSpark, quantization
scales, and the fixed E2M1 codebook.

Successful metadata binding joins authenticated content with the canonical
advanced execution-plan digest. Host/device materialization still returns an
explicit unsupported-byte-layout result; recognizing FP8 or packed FP4 metadata
does not grant decode, allocation, device, kernel, or execution authority.
