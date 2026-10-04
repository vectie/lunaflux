///|
test "streamed startup supports nibble offsets and scale grids without a tensor-sized allocation" {
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
    let grouping = if format == Int4 {
      Some(@format.ScaleGrouping::Block(rows=2, columns=7))
    } else {
      None
    }
    let rep = @format.Representation::new(
      format~,
      rows=3,
      columns=35,
      grouping?,
    )
    let source = @format.quantize(
      rep,
      Array::makei(105, i => Float::from_int(i % 19 - 9) * 0.25F)[:],
    )
    let plan = @precision.plan_operand(
      role=Weight,
      representation=rep,
      placement=StartupExpandBf16,
      supported_placements=[StartupExpandBf16],
      accuracy=@precision.Accuracy::new(absolute=0.2, relative=0.1),
      memory_ceiling=210L,
    )
    let actual = FixedArray::make(210, b'\xaa')
    let mut largest_read = 0
    @precision_materialize.stream_bf16_weight(
      plan,
      global_scale=source.global_scale(),
      chunk_elements=7,
      host_scratch_ceiling=128L,
      (plane, offset, count) => {
        largest_read = largest_read.max(count)
        let bytes = match plane {
          Payload => source.payload()
          Scales => source.scales()
        }
        bytes.sub(start=offset.to_int(), end=offset.to_int() + count).to_owned()
      },
      (offset, bytes) => {
        for i, b in bytes {
          actual[offset.to_int() + i] = b
        }
      },
    )
    assert_true(largest_read <= 28)
    let expected = FixedArray::make(210, b'\x00')
    source.decode_bf16_chunk(0L, 105, expected)
    assert_eq(actual, expected)
  }
}
