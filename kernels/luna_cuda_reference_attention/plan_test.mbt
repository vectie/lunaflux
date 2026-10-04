///|
fn fixture(metadata? : Bool = false) -> @abi.CudaPagedAttentionTileAbi raise {
  @abi.CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=9216L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=9216,
    query_metadata=metadata,
  )
}

///|
test "reference lowerings retain distinct numerical and launch contracts" {
  let backends : Array[@luna_cuda_reference_attention.ReferenceAttentionBackend] = [
    FlashAttentionPrefill,
    FlashAttentionDecode,
    FlashInferDecode,
  ]
  for backend in backends {
    let plan = @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
      backend~,
      abi=fixture(),
    )
    let source = @utf8.decode(plan.source())
    assert_true(source.contains(plan.symbol()))
    assert_true(!source.contains("cudaMalloc"))
    assert_true(!source.contains("ATen"))
    assert_eq(plan.source(), plan.source())
    assert_eq(plan.source_sha256().length(), 64)
    let (_, _, _, x, y, z, _) = plan.launch()
    assert_eq(x * y * z, 128)
  }
}

///|
test "query metadata ABI cannot be silently reinterpreted" {
  try
    @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
      backend=FlashInferDecode,
      abi=fixture(metadata=true),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("incompatible query metadata admitted")
  }
}

///|
test "BF16 operands and F32 accumulator are distinct from probability precision" {
  for
    backend in [
      @luna_cuda_reference_attention.ReferenceAttentionBackend::FlashAttentionPrefill,
      FlashInferDecode,
    ] {
    let plan = @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
      backend~,
      abi=fixture(),
      query_dtype=@numeric.ComputeDType::bfloat16(),
      kv_dtype=@numeric.ComputeDType::bfloat16(),
      output_dtype=@numeric.ComputeDType::bfloat16(),
      accumulator=@numeric.AccumulatorDType::f32(),
    )
    let (query, kv, output, accumulator) = plan.operand_formats()
    assert_eq(query, @numeric.ComputeDType::bfloat16())
    assert_eq(kv, query)
    assert_eq(output, query)
    assert_eq(accumulator, @numeric.AccumulatorDType::f32())
    let source = @utf8.decode(plan.source())
    assert_true(source.contains("const __nv_bfloat16 *input"))
  }
}

///|
test "quantized KV or queries cannot be silently cast to BF16" {
  for
    dtype in [
      @numeric.ComputeDType::f8_e4m3_finite(),
      @numeric.ComputeDType::i8(),
    ] {
    try
      @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
        backend=FlashInferDecode,
        abi=fixture(),
        kv_dtype=dtype,
      )
    catch {
      @luna_cuda_reference_attention.UnsupportedReferenceAttentionFormat => ()
      _ => fail("wrong format failure")
    } noraise {
      _ => fail("quantized cache interpreted as BF16")
    }
    try
      @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
        backend=FlashAttentionPrefill,
        abi=fixture(),
        query_dtype=dtype,
      )
    catch {
      @luna_cuda_reference_attention.UnsupportedReferenceAttentionFormat => ()
      _ => fail("wrong format failure")
    } noraise {
      _ => fail("quantized queries interpreted as BF16")
    }
  }
}

///|
test "reference decode reuses grouped KV without rewriting model semantics" {
  let plan = @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
    backend=FlashAttentionDecode,
    abi=fixture(),
  )
  let (_, heads, _, _, _, _, _) = plan.launch()
  assert_eq(heads, 8)
  let source = @utf8.decode(plan.source())
  assert_true(source.contains("p.seqlen_q=2"))
  assert_true(source.contains("p.q_row_stride=128"))
  assert_true(source.contains("p.is_causal=false"))
}
