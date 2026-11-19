///|
/// Reuses the existing paged-attention compiler and BF16 KV ABI. This object
/// contains generated sources and plans, not nvcc output or device evidence.
pub struct Compilation {
  priv prefill : @aot.CudaAttentionTileAotCompilation
  priv decode : @aot.CudaAttentionTileAotCompilation
  priv abi : @source.CudaPagedAttentionTileAbi
  priv context_tokens : Int32
  priv cache_bytes : Int64
}

///|
pub fn compile(
  operators : @operators.DecoderOperators,
  device_target~ : @catalog.DeviceTarget,
  target_id~ : @strategy.AttentionTargetId,
  capabilities~ : @strategy.AttentionCompilerCapabilities,
  maximum_context_tokens~ : Int32 = 4096,
  maximum_sequences~ : Int32 = 1,
  tokens_per_page~ : Int32 = 16,
  autotune_records~ : ArrayView[@strategy.AttentionAutotuneRecord] = [],
) -> Compilation raise {
  let (query_heads, width, _, rows, maximum_positions) = operators
    .query_rotary()
    .geometry()
  let (kv_heads, _, _, _, _) = operators.key_rotary().geometry()
  guard maximum_context_tokens > 0 &&
    maximum_context_tokens <= maximum_positions &&
    maximum_sequences > 0 &&
    maximum_sequences <= rows else {
    fail("invalid MiniCPM attention capacity")
  }
  let shape = @strategy.AttentionShape::new(
    query_heads~,
    kv_heads~,
    head_dimension=width,
    tokens_per_page~,
  )
  let pages = (
      maximum_context_tokens.to_int64() + tokens_per_page.to_int64() - 1L
    ) /
    tokens_per_page.to_int64() *
    maximum_sequences.to_int64()
  guard pages <= 2147483647L else {
    fail("attention page-table capacity overflow")
  }
  let stride = tokens_per_page.to_int64() *
    kv_heads.to_int64() *
    width.to_int64()
  guard stride <= 2305843009213693951L / pages else {
    fail("attention cache capacity overflow")
  }
  let abi = @source.CudaPagedAttentionTileAbi::new(
    query_heads~,
    key_value_heads=kv_heads,
    head_dimension=width,
    tokens_per_page~,
    input_row_width=(query_heads + kv_heads * 2) * width,
    page_stride_values=stride,
    physical_page_capacity=pages,
    maximum_query_rows=maximum_sequences,
    maximum_query_tokens=rows,
    maximum_page_entries=pages.to_int(),
  )
  let build = fn(phase, name) raise {
    let problem = @strategy.AttentionCompileProblem::new(
      shape~,
      phase~,
      numeric_type=AttentionBFloat16,
      batch_rows=maximum_sequences,
      query_rows=if phase is Decode { 1 } else { rows },
      context_tokens=maximum_context_tokens,
      causal=true,
      key_value_read_view=PagedCacheOnly,
    )
    let (_, shared_memory, _) = capabilities.limits()
    @aot.compile_cuda_attention_tile_aot(
      @aot.CudaAttentionTileAotRequest::new(
        backend=@strategy.AttentionBackendId::new("cuda"),
        target_id~,
        problem~,
        capabilities~,
        device_target~,
        maximum_shared_memory_bytes=shared_memory,
        abi~,
        symbol=name,
        maximum_source_bytes=2097152,
      ),
      autotune_records,
    ) catch {
      @aot.CudaAttentionTileAotError::InvalidCudaAttentionTileAot(reason) =>
        fail("MiniCPM attention \{Repr(phase)}: \{Repr(reason)}")
      error => raise error
    }
  }
  {
    prefill: build(Prefill, "lunaflux_minicpm_v47_attention_prefill"),
    decode: build(Decode, "lunaflux_minicpm_v47_attention_decode"),
    abi,
    context_tokens: maximum_context_tokens,
    cache_bytes: pages * stride * 4L,
  }
}

///|
pub fn Compilation::prefill(
  self : Self,
) -> @aot.CudaAttentionTileAotCompilation {
  self.prefill
}

///|
pub fn Compilation::decode(self : Self) -> @aot.CudaAttentionTileAotCompilation {
  self.decode
}

///|
pub fn Compilation::abi(self : Self) -> @source.CudaPagedAttentionTileAbi {
  self.abi
}

///|
pub fn Compilation::context_tokens(self : Self) -> Int32 {
  self.context_tokens
}

///|
/// Both K and V planes, including page padding, per full-attention layer.
pub fn Compilation::cache_bytes(self : Self) -> Int64 {
  self.cache_bytes
}
