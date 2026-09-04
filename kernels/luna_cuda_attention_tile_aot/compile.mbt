///|
/// Lowers one already-scheduled functional variant through the CUDA-only
/// terminal passes. Keeping this function explicit lets a functional frontier
/// share the same lowering without rebuilding or mutating compiler state.
fn lower_cuda_attention_tile_aot_variant(
  request : CudaAttentionTileAotRequest,
  functional : @luna_attention_tile_compiler.ScheduledAttentionTileCompilation,
  symbol : String,
) -> CudaAttentionTileAotCompilation raise CudaAttentionTileAotError {
  let lowering = @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    functional.program(),
    functional.optimization(),
    functional.schedule(),
    request.device_target,
    request.maximum_shared_memory_bytes,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(DeviceLowering)
  }
  let source = @luna_cuda_attention_tile_source.emit_cuda_attention_tile_source(
    lowering,
    request.abi,
    symbol,
    request.maximum_source_bytes,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(SourceEmission)
  }
  let canonical = @utf8.encode(
    "luna-cuda-attention-tile-aot-v1\n" +
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
/// Composes the pure functional compiler with the CUDA-only terminal passes.
/// The function performs no I/O and has no mutable compiler context.
pub fn compile_cuda_attention_tile_aot(
  request : CudaAttentionTileAotRequest,
  autotune_records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> CudaAttentionTileAotCompilation raise CudaAttentionTileAotError {
  let functional = @luna_attention_tile_compiler.compile_attention_tile(
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
  lower_cuda_attention_tile_aot_variant(request, functional, request.symbol)
}

///|
/// Lowers every non-dominated functional variant to a distinct AOT CUDA
/// symbol. This is a backend terminal map over an immutable functional value;
/// candidate discovery, dominance, and selection stay outside CUDA.
pub fn compile_cuda_attention_tile_aot_frontier(
  request : CudaAttentionTileAotRequest,
  autotune_records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> CudaAttentionTileAotFrontier raise CudaAttentionTileAotError {
  let functional = @luna_attention_tile_compiler.compile_attention_tile_frontier(
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
  let variants : Array[CudaAttentionTileAotCompilation] = []
  for compilation in functional.variants() {
    let stable_id = compilation.plan().candidate().stable_id()
    // Preserve the long-lived production ABI symbol for the selected variant.
    // Non-selected frontier members receive stable candidate suffixes so a
    // bundle can retain them without breaking existing selected-only users.
    let symbol = if compilation == functional.selected() {
      request.symbol
    } else {
      request.symbol + "_c" + stable_id.to_string()
    }
    try
      variants.push(
        lower_cuda_attention_tile_aot_variant(request, compilation, symbol),
      )
    catch {
      _ => ()
    } noraise {
      _ => ()
    }
  }
  if variants.length() == 0 {
    raise InvalidCudaAttentionTileAot(SourceEmission)
  }
  guard variants.filter(variant => variant.functional() == functional.selected())
    is [selected] else {
    raise InvalidCudaAttentionTileAot(FunctionalCompilation)
  }
  let builder = StringBuilder()
  builder <+ "luna-cuda-attention-tile-aot-frontier-v1\n"
  builder <+ "functional_sha256=\{functional.digest().as_hex()}\n"
  builder <+ "selected=\{selected.digest().as_hex()}\n"
  builder <+ "variants=\{variants.length()}\n"
  for variant in variants {
    builder <+
      "variant=\{variant.functional().plan().candidate().stable_id()},\{variant.source().symbol()},\{variant.digest().as_hex()}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  {
    functional,
    selected,
    variants: ReadOnlyArray::from_array(variants),
    canonical,
    digest: CudaAttentionTileAotDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
