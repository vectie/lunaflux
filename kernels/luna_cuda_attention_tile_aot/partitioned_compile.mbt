///|
/// Purely composes the functional partitioned compiler with CUDA lowering and
/// deterministic two-entry-point source emission.
pub fn compile_cuda_partitioned_attention_tile_aot(
  request : CudaPartitionedAttentionTileAotRequest,
  autotune_records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> CudaPartitionedAttentionTileAotCompilation raise CudaAttentionTileAotError {
  let functional = @luna_attention_tile_compiler.compile_partitioned_attention_tile(
    @luna_attention_tile_compiler.AttentionTileCompilerRequest::new(
      backend=request.backend,
      target=request.target_id,
      problem=request.problem,
      capabilities=request.capabilities,
      autotune_records,
    ),
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(FunctionalCompilation)
  }
  let lowering = @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    functional.program(),
    functional.optimization(),
    functional.schedule(),
    request.device_target,
    request.maximum_shared_memory_bytes,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(DeviceLowering)
  }
  let source = @luna_cuda_attention_tile_source.emit_cuda_partitioned_attention_tile_source(
    lowering,
    request.abi,
    request.partial_symbol,
    request.merge_symbol,
    request.maximum_workspace_bytes,
    request.maximum_source_bytes,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(SourceEmission)
  }
  let canonical = @utf8.encode(
    "luna-cuda-partitioned-attention-tile-aot-v1\n" +
    "functional_sha256=" +
    functional.digest().as_hex() +
    "\n" +
    "lowering_sha256=" +
    lowering.digest().as_hex() +
    "\n" +
    "source_sha256=" +
    source.digest().as_hex() +
    "\n",
  )
  {
    functional,
    lowering,
    source,
    canonical,
    digest: CudaAttentionTileAotDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
