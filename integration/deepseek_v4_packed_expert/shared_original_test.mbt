///|
async test "literal shared expert replay exports actual production FP8 plan" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let matrix = @env.get_env_var("LUNAFLUX_MATCHED_SHARED_MATRIX") == Some("1")
  let plan = @deepseek_v4_packed_expert.shared_expert_compute_plan(
    spec,
    layer_index=0,
    maximum_rows=1,
    memory_ceiling=268435456L,
    accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
    scaled_dot_law=if matrix {
      UnscaledBlockMatrixFloat32(
        block_columns=128,
        publication=SeparateMultiplyAdd,
      )
    } else {
      DecodeOperandsFloat32
    },
  )
  let source = @expert_source.packed_expert_mlp_source(
    plan,
    symbol="matched_shared",
  )
  assert_eq(source.launches()[1], (if matrix { 64 } else { 512 }, 128))
  assert_eq(source.launches()[3], (if matrix { 128 } else { 1024 }, 128))
  assert_eq(plan.weights()[0].representation().scale_shape(), (16, 32))
  let offsets = plan.weight_offsets()
  let sizes = plan.quantized_workspace_bytes()
  let metadata = @utf8.encode(
    "\nstatic constexpr size_t shared_stride=\{plan.expert_stride()}ULL;\n" +
    "static constexpr size_t shared_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};\n" +
    "static constexpr size_t shared_quantized_bytes[2]={\{sizes.0}ULL,\{sizes.1}ULL};\n" +
    "#define LUNA_SHARED_LAUNCH_METADATA 1\nstatic constexpr int shared_gate_grid=\{source.launches()[1].0},shared_down_grid=\{source.launches()[3].0};\n",
  )
  if @env.get_env_var("LUNAFLUX_MATCHED_SHARED_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(
      path,
      source.source_bytes() + metadata,
      create_mode=CreateNew,
    )
  }
}
