///|
fn prefix_fixture() -> @precision.QuantizedQueryKvPrecision raise {
  @precision.QuantizedQueryKvPrecision::new(
    hidden_width=256,
    query_rank=128,
    head_count=2,
    head_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=6144L,
  )
}

///|
test "quantized query prefix composes unique entrypoints and shared FP8 helpers" {
  let source = @quantized_query_kv_cuda_source.quantized_query_kv_source(
    prefix_fixture(),
    symbol_prefix="query_fixture",
  )
  for i in 0..<6 {
    assert_true(@utf8.decode(source.source_bytes()).contains(source.symbol(i)))
  }
  assert_eq(source.launch(2), (3, 1, 256))
  assert_eq(source.launch(3), (3, 2, 256))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("#ifndef LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1"))
  assert_true(text.contains("__float2bfloat16_rn(__fmul_rn(value, value))"))
  assert_true(text.contains("__float2bfloat16_rn(__fadd_rn(mean,"))
  assert_true(text.contains("__float2bfloat16_rn(rsqrtf(variance))"))
}

///|
test "unimplemented quantization policy is not silently substituted" {
  let plan = @precision.QuantizedQueryKvPrecision::new(
    hidden_width=256,
    query_rank=128,
    head_count=2,
    head_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0,
    maximum_workspace_bytes=6144L,
  )
  try
    @quantized_query_kv_cuda_source.quantized_query_kv_source(
      plan,
      symbol_prefix="query",
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    error => raise error
  } noraise {
    _ => fail("different amax policy silently accepted")
  }
}

///|
test "offline quantized query prefix exporter" {
  if @env.get_env_var("LUNA_EXPORT_QUANTIZED_QUERY_KV") == Some("1") {
    println("LUNA_QUERY_KV_SOURCE_BEGIN")
    println(
      @utf8.decode(
        @quantized_query_kv_cuda_source.quantized_query_kv_source(
          prefix_fixture(),
          symbol_prefix="query_fixture",
        ).source_bytes(),
      ),
    )
    println("LUNA_QUERY_KV_SOURCE_END")
  }
}

///|
async test "original DeepSeek query replay exports full-width numerical stages" {
  let matrix = @env.get_env_var("LUNAFLUX_MATCHED_QUERY_MATRIX") == Some("1")
  let plan = @precision.QuantizedQueryKvPrecision::new(
    hidden_width=4096,
    query_rank=1024,
    head_count=64,
    head_width=512,
    maximum_rows=144,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=33554432L,
    query_square_law=VectorizedTreeFloat32(
      partitions=256,
      vector_width=if matrix { 4 } else { 1 },
    ),
    projection_scaled_dot_law=if matrix {
      UnscaledBlockMatrixFloat32(
        block_columns=128,
        publication=SeparateMultiplyAdd,
      )
    } else {
      DecodeOperandsFloat32
    },
  )
  let source = @quantized_query_kv_cuda_source.quantized_query_kv_source(
    plan,
    symbol_prefix="matched_query",
  )
  assert_eq(
    source.launch(2),
    if matrix {
      (144, 1024, 128)
    } else {
      (144, 128, 256)
    },
  )
  assert_eq(source.launch(3), (144, 64, 256))
  if !matrix &&
    @env.get_env_var("LUNAFLUX_MATCHED_QUERY_PREVIOUS_SOURCE") is Some(path) {
    let old = @fs.read_file(path)
      .text()
      .split("\n#define LUNA_MATCHED_QUERY_NORM_VARIANTS")
      .collect()[0]
    assert_eq(@utf8.decode(source.source_bytes()), old.to_owned())
  }
  if @env.get_env_var("LUNAFLUX_MATCHED_QUERY_SOURCE_OUTPUT") is Some(path) {
    let mut bytes = source.source_bytes() +
      b"\n#define LUNA_MATCHED_QUERY_NORM_VARIANTS 1\n"
    for stage in 0..<6 {
      let (_, y, threads) = source.launch(stage)
      bytes += @utf8.encode(
        "#define LUNA_QUERY_Y_\{stage} \{y}\n#define LUNA_QUERY_THREADS_\{stage} \{threads}\n",
      )
    }
    for
      (partitions, vector) in [
        (128, 4),
        (256, 2),
        (256, 4),
        (512, 1),
        (512, 2),
        (512, 4),
      ] {
      let norm = @norm.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
        prefix_width=1024,
        suffix_width=0,
        maximum_rows=144,
        epsilon=1.0e-6,
        function_symbol="matched_norm_\{partitions}_\{vector}",
        square_law=VectorizedTreeFloat32(partitions~, vector_width=vector),
      )
      assert_eq(norm.launch_threads(), partitions)
      bytes += norm.source_bytes() + b"\n"
    }
    @fs.write_file(path, bytes, create_mode=CreateNew)
  }
}

///|
async test "matched output B exports independent raw projection arithmetic" {
  let source = @projection.block_fp8_ue8m0_projection_cuda_source(
    input_width=8192,
    output_width=4096,
    maximum_rows=144,
    function_symbol="matched_projection",
    scaled_dot_law=UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd,
    ),
  )
  let (_, y, threads) = source.launch()
  assert_eq((y, threads), (128, 128))
  if @env.get_env_var("LUNAFLUX_MATCHED_OUTPUT_B_MATRIX_SOURCE_OUTPUT")
    is Some(path) {
    @fs.write_file(
      path,
      source.source_bytes() +
      @utf8.encode(
        "\n#define LUNA_PROJECTION_Y \{y}\n#define LUNA_PROJECTION_THREADS \{threads}\n",
      ),
      create_mode=CreateNew,
    )
  }
}
