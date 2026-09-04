///|
fn finish_cuda_partitioned_attention_tile_aot(
  request : CudaPartitionedAttentionTileAotRequest,
  functional : @luna_attention_tile_compiler.ScheduledAttentionTileCompilation,
  partial_symbol : String,
  merge_symbol : String,
) -> CudaPartitionedAttentionTileAotCompilation raise CudaAttentionTileAotError {
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
    partial_symbol,
    merge_symbol,
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
  finish_cuda_partitioned_attention_tile_aot(
    request,
    functional,
    request.partial_symbol,
    request.merge_symbol,
  )
}

///|
/// Lowers every distinct functional partition schedule into an immutable CUDA
/// source pair. The first variant retains the requested ABI symbols; later
/// variants receive a stable `_pN` suffix derived only from partition count.
pub fn compile_cuda_partitioned_attention_tile_aot_frontier(
  request : CudaPartitionedAttentionTileAotRequest,
  autotune_records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
  parallel_workgroup_targets : ArrayView[Int],
) -> CudaPartitionedAttentionTileAotFrontier raise CudaAttentionTileAotError {
  let functional = @luna_attention_tile_compiler.compile_partitioned_attention_tile_frontier(
    @luna_attention_tile_compiler.AttentionTileCompilerRequest::new(
      backend=request.backend,
      target=request.target_id,
      problem=request.problem,
      capabilities=request.capabilities,
      autotune_records,
    ),
    parallel_workgroup_targets,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(FunctionalCompilation)
  }
  let variants : Array[CudaPartitionedAttentionTileAotVariant] = []
  for index, variant in functional.variants() {
    let partitions = variant.compilation().schedule().key_value_partitions()
    let suffix = if index == 0 { "" } else { "_p" + partitions.to_string() }
    let compilation = finish_cuda_partitioned_attention_tile_aot(
      request,
      variant.compilation(),
      request.partial_symbol + suffix,
      request.merge_symbol + suffix,
    )
    variants.push({
      parallel_workgroup_target: variant.parallel_workgroup_target(),
      compilation,
    })
  }
  let builder = StringBuilder()
  builder <+ "luna-cuda-partitioned-attention-frontier-v1\n"
  builder <+ "functional_sha256=\{functional.digest().as_hex()}\n"
  builder <+ "variants=\{variants.length()}\n"
  for variant in variants {
    let partitions = variant.compilation
      .functional()
      .schedule()
      .key_value_partitions()
    builder <+
      "variant=\{variant.parallel_workgroup_target},\{partitions},\{variant.compilation.digest().as_hex()}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  {
    functional,
    variants: ReadOnlyArray::from_array(variants),
    canonical,
    digest: CudaAttentionTileAotDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
