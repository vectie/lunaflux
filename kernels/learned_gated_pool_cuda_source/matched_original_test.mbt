///|
async test "original pool replay covers full ratio4 and ratio128 projection and state geometry" {
  for ratio in [4, 128] {
    let plan = @precision.LearnedGatedPoolPrecision::new(
      input_width=4096,
      feature_width=512,
      ratio~,
      preceding_window_overlap=ratio == 4,
      maximum_rows=144,
      maximum_positions=256,
      norm_epsilon=0.000001,
      maximum_state_bytes=16777216L,
      maximum_workspace_bytes=16777216L,
    )
    let source = @learned_gated_pool_cuda_source.learned_gated_pool_source(
      plan,
      symbol_prefix="matched_pool",
    )
    let cache = @rotary_source.rotary_cache_source(
      @precision.RotaryCachePrecision::new(
        head_count=1,
        head_width=512,
        rotary_width=64,
        maximum_rows=(144 + ratio - 1) / ratio,
        maximum_positions=1048576,
        frequency=Yarn(
          theta=160000.0,
          original_positions=65536,
          factor=16.0,
          beta_fast=32.0,
          beta_slow=1.0,
        ),
        simulation_block_width=64,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes=16777216L,
      ),
      symbol_prefix="matched_cache",
    )
    assert_eq(cache.launch(1), ((144 + ratio - 1) / ratio, 7, 64))
    assert_eq(source.launch(1), (144, if ratio == 4 { 4 } else { 2 }, 256))
    let channels = if ratio == 4 { 1024 } else { 512 }
    let metadata = @utf8.encode(
      "\nstatic constexpr int pool_ratio=\{ratio},pool_channels=\{channels},pool_state_rows=\{if ratio == 4 { 8 } else { 128 }};\n",
    )
    if @env.get_env_var("LUNAFLUX_MATCHED_POOL_SOURCE_OUTPUT_\{ratio}")
      is Some(path) {
      @fs.write_file(
        path,
        source.source_bytes() + b"\n" + cache.source_bytes() + metadata,
        create_mode=CreateNew,
      )
    }
  }
}
