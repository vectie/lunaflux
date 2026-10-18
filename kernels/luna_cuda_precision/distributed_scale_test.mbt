///|
test "distributed scale export preserves numeric extent and two-launch contract" {
  for multiplier in [1, 8] {
    let width = if multiplier == 1 { 4096 } else { 1024 }
    let q = @precision.plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=NvFp4,
        rows=64 * multiplier,
        columns=width,
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=16777216L,
    ).quantization()
    let bytes = @precision.packed_buffer_layout(q.operand().representation()).buffer_bytes()
    let old = @luna_cuda_precision.quantize_fp4_bf16_rows_source(
      q,
      row_multiplier=multiplier,
      symbol="baseline",
    )
    for groups in [16, 32, 64] {
      let p = @physical.PartitionedMaximum::new(
        partitions=groups,
        group_elements=256,
        after_bytes=bytes,
      ).unwrap()
      let fresh = @luna_cuda_precision.quantize_fp4_bf16_rows_source(
        q,
        row_multiplier=multiplier,
        symbol="candidate",
        maximum=p,
      )
      assert_true(fresh.contains("producers*256"))
      assert_true(fresh.contains("if(cell==0)"))
      assert_true(p.partial_offset() >= bytes)
      if @env.get_env_var("LUNA_EXPORT_DISTRIBUTED_SCALE") == Some("1") {
        println("LUNA_SOURCE_BEGIN scale-\{multiplier}-\{groups}")
        println(old + "\n" + fresh)
        println("constexpr int R=64,H=\{width},S=\{multiplier},G=\{groups};")
        println(
          "constexpr size_t NUMERIC=\{bytes},STORAGE=\{p.allocation_bytes()};",
        )
        println("LUNA_SOURCE_END")
      }
    }
  }
}
