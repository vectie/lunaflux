///|
test "offline rotary Hadamard FP4 numerical exporter" {
  if @env.get_env_var("LUNA_EXPORT_HADAMARD_FP4") != Some("1") {
    return
  }
  let plan = @precision.RotaryHadamardFp4Precision::new(
    head_count=2,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1024,
    frequency=Base(theta=10000.0),
    simulation_block_width=32,
    maximum_workspace_bytes=3072L,
  )
  println("LUNA_HADAMARD_FP4_SOURCE_BEGIN")
  println(
    @utf8.decode(
      @rotary_hadamard_fp4_cuda_source.rotary_hadamard_fp4_source(
        plan,
        symbol_prefix="hadamard_fp4",
      ).source_bytes(),
    ),
  )
  println("LUNA_HADAMARD_FP4_SOURCE_END")
}
