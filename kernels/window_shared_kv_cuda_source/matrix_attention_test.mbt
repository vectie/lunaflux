///|
fn matrix_plan(
  heads : Int,
  width : Int,
  accumulation : @precision.AttentionAccumulationLaw,
) -> @precision.WindowSharedKvPrecision raise {
  @precision.WindowSharedKvPrecision::new(
    query_heads=heads,
    head_width=width,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=1048576,
    maximum_state_bytes=33554432L,
    probability=OnlineBf16(block_tokens=64),
    accumulation~,
  )
}

///|
test "explicit tiled arithmetic propagates to source and launch ownership" {
  let plan = matrix_plan(18, 80, TiledFloat32)
  let source = @window_shared_kv_cuda_source.window_shared_kv_source(
    plan,
    symbol_prefix="matrix",
  )
  assert_eq(source.plan().accumulation_law(), TiledFloat32)
  assert_eq(source.launch(1), (144, 2, 256))
  assert_eq(source.launch(2), (128, 1, 128))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("wm::mma_sync"))
  assert_true(text.contains("__shfl_xor_sync"))
  assert_true(text.contains("accumulated[2]"))
  assert_true(text.contains("key&&col<80"))
  assert_true(text.contains("if(h<18 && col<80)"))
  assert_true(text.contains("absolute % 128"))
  assert_true(text.contains("__float2bfloat16_rn(p)"))
  assert_true(text.contains("maxima[h]==-INFINITY?0.0f"))
}

///|
test "default and explicit ordered source remain byte identical" {
  let omitted = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=512,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=1048576,
    maximum_state_bytes=33554432L,
    probability=OnlineBf16(block_tokens=64),
  )
  let explicit = matrix_plan(2, 512, OrderedFloat32)
  assert_eq(
    @window_shared_kv_cuda_source.window_shared_kv_source(
      omitted,
      symbol_prefix="same",
    ).source_bytes(),
    @window_shared_kv_cuda_source.window_shared_kv_source(
      explicit,
      symbol_prefix="same",
    ).source_bytes(),
  )
  assert_eq(
    @window_shared_kv_cuda_source.window_shared_kv_source(
      explicit,
      symbol_prefix="same",
    ).launch(1),
    (144, 2, 128),
  )
}

///|
test "unsupported matrix geometry never silently substitutes scalar arithmetic" {
  try
    @window_shared_kv_cuda_source.window_shared_kv_source(
      matrix_plan(2, 24, TiledFloat32),
      symbol_prefix="bad",
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    _ => fail("wrong unsupported matrix error")
  } noraise {
    _ => fail("unsupported matrix geometry accepted")
  }
}

///|
async test "offline matrix window exporter uses the retained numerical policy" {
  if @env.get_env_var("LUNAFLUX_MATRIX_WINDOW_SOURCE_OUTPUT") is Some(path) {
    let source = @window_shared_kv_cuda_source.window_shared_kv_source(
      matrix_plan(64, 512, TiledFloat32),
      symbol_prefix="matched_window",
    )
    assert_eq(source.launch(1), (144, 4, 256))
    let tail = @window_shared_kv_cuda_source.window_shared_kv_source(
      matrix_plan(18, 80, TiledFloat32),
      symbol_prefix="tail_window",
    )
    let (_, grid_y, threads) = source.launch(1)
    let (_, tail_grid_y, tail_threads) = tail.launch(1)
    @fs.write_file(
      path,
      @utf8.encode(
        "#define MATRIX_GRID_Y \{grid_y}\n#define MATRIX_THREADS \{threads}\n#define TAIL_GRID_Y \{tail_grid_y}\n#define TAIL_THREADS \{tail_threads}\n",
      ) +
      source.source_bytes() +
      b"\n" +
      tail.source_bytes(),
      create_mode=CreateNew,
    )
  }
}
