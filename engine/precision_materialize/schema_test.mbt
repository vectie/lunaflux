///|
test "mixed-format graph weight schema streams every tensor into one BF16 arena" {
  let formats = [
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
  ]
  let sources = formats.map(format => {
    @format.quantize(
      @format.Representation::new(format~, rows=3, columns=35),
      Array::makei(105, i => Float::from_int(i % 19 - 9) * 0.25F)[:],
    )
  })
  let operands = sources.map(source => {
    @precision.plan_operand(
      role=Weight,
      representation=source.representation(),
      placement=StartupExpandBf16,
      supported_placements=[StartupExpandBf16],
      accuracy=@precision.Accuracy::new(absolute=0.2, relative=0.1),
      memory_ceiling=210L,
    )
  })
  let schema = @precision.WeightSchema::new(
    operands[:],
    alignment_bytes=64L,
    scratch_lifetime=Sequential,
    device_memory_ceiling=4096L,
  )
  let (resident, scratch) = schema.memory()
  assert_eq(scratch, 0L)
  let actual = FixedArray::make(resident.to_int(), b'\xaa')
  let expected = FixedArray::make(resident.to_int(), b'\xaa')
  for binding in schema.bindings() {
    sources[binding.ordinal()].decode_bf16_chunk(
      0L,
      105,
      expected,
      destination_offset=binding.offset().to_int(),
    )
  }
  @precision_materialize.stream_bf16_weights(
    schema,
    sources.map(source => source.global_scale())[:],
    chunk_elements=7,
    host_scratch_ceiling=128L,
    (ordinal, plane, offset, count) => {
      let source = sources[ordinal]
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
  assert_eq(actual, expected)
}

///|
test "mixed route and invalid global scale fail before any startup effect" {
  let representation = @format.Representation::new(
    format=NvFp4,
    rows=1,
    columns=16,
  )
  let accuracy = @precision.Accuracy::new(absolute=0.1, relative=0.1)
  let expand = @precision.plan_operand(
    role=Weight,
    representation~,
    placement=StartupExpandBf16,
    supported_placements=[StartupExpandBf16],
    accuracy~,
    memory_ceiling=1000L,
  )
  let compact = @precision.plan_operand(
    role=Weight,
    representation~,
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy~,
    memory_ceiling=1000L,
    tile_elements=16,
  )
  let mut effects = 0
  for
    (plans, globals) in [
      ([expand, compact], [1.0F, 1.0F]),
      ([expand, expand], [1.0F, 0.0F]),
    ] {
    let schema = @precision.WeightSchema::new(
      plans[:],
      alignment_bytes=64L,
      scratch_lifetime=Sequential,
      device_memory_ceiling=1000L,
    )
    try
      @precision_materialize.stream_bf16_weights(
        schema,
        globals[:],
        chunk_elements=7,
        host_scratch_ceiling=128L,
        (_, _, _, _) => {
          effects = effects + 1
          b""
        },
        (_, _) => effects = effects + 1,
      )
    catch {
      WrongPlan | Conversion(_) => ()
    } noraise {
      _ => fail("invalid startup schema was executed")
    }
  }
  assert_eq(effects, 0)
}
