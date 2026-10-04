///|
test "bounded startup materialization all formats joins unchanged BF16 compute" {
  for
    format in [
      @format.Format::Bf16,
      Fp16,
      F32,
      Fp8E4M3,
      Fp8E5M2,
      Int8,
      Int4,
      Fp4E2M1,
      NvFp4,
      MxFp4,
      MxFp8E4M3,
      MxFp8E5M2,
    ] {
    let rep = @format.Representation::new(format~, rows=3, columns=35)
    let source = @format.quantize(
      rep,
      Array::makei(105, i => Float::from_int(i % 19 - 9) * 0.25F)[:],
    )
    let plan = @precision.plan_operand(
      role=Weight,
      representation=rep,
      placement=StartupExpandBf16,
      supported_placements=[StartupExpandBf16],
      accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=210L,
    )
    let materialized = @precision_materialize.StartupWeight::new(plan, source)
    assert_eq(materialized.bytes(), 210L)
    let arena = FixedArray::make(212, b'\xaa')
    for first = 0; first < 105; first = first + 7 {
      materialized.write_chunk(first.to_int64(), 7, arena, offset=1 + first * 2)
    }
    assert_eq(arena[0], b'\xaa')
    assert_eq(arena[211], b'\xaa')
    for i in 0..<105 {
      let code = arena[1 + i * 2].to_uint() | (arena[2 + i * 2].to_uint() << 8)
      assert_eq(
        code,
        @format.encode_element(Bf16, source.decode(i / 35, i % 35)),
      )
    }
  }
}
