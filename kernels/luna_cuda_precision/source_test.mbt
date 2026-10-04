///|
test "AOT kernels distinguish scale stages and reject implicit serving fallback" {
  let accuracy = @precision.Accuracy::new(absolute=0.1, relative=0.1)
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
    let rep = @format.Representation::new(format~, rows=2, columns=33)
    let decode = @precision.plan_operand(
      role=Weight,
      representation=rep,
      placement=StartupExpandBf16,
      supported_placements=[StartupExpandBf16],
      accuracy~,
      memory_ceiling=1000L,
    )
    let code = @luna_cuda_precision.decode_source(
      decode,
      symbol="decode_weight",
    )
    assert_true(code.source().contains("i >= 66ULL"))
    assert_eq(code.launch().dimensions(), (1, 256))
    assert_eq(
      code.digest(),
      @luna_cuda_precision.decode_source(decode, symbol="decode_weight").digest(),
    )
    let quant = @precision.plan_operand(
      role=Activation,
      representation=rep,
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy~,
      memory_ceiling=1000L,
    )
    let source = @luna_cuda_precision.quantize_source(
      quant,
      symbol="pack_activation",
    )
    assert_eq(
      source.stages().last().unwrap(),
      @luna_cuda_precision.QuantizationStage::Pack,
    )
    assert_eq(source.stages().contains(GlobalScale), format == NvFp4)
    assert_eq(source.stages().length(), source.launches().length())
    for launch in source.launches() {
      assert_eq(launch.dimensions().1, 256)
    }
    assert_true(!source.source().contains("\\{symbol}"))
  }
}

///|
test "launch limits are checked before exposing executable kernels" {
  let rep = @format.Representation::new(
    format=MxFp4,
    rows=2147483647,
    columns=128,
  )
  let plan = @precision.plan_operand(
    role=Activation,
    representation=rep,
    placement=QuantizeBeforeConsumer,
    supported_placements=[QuantizeBeforeConsumer],
    accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
    memory_ceiling=200000000000L,
  )
  try
    @luna_cuda_precision.quantize_source(plan, symbol="too_many_scale_blocks")
  catch {
    UnsupportedGeometry => ()
    _ => fail("wrong launch failure")
  } noraise {
    _ => fail("scale grid overflow escaped lowering")
  }
}

///|
test "a tile scratch plan cannot lower to a whole-tensor conversion kernel" {
  let tile = @precision.plan_operand(
    role=Weight,
    representation=@format.Representation::new(
      format=Int4,
      rows=1024,
      columns=1024,
    ),
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
    tile_elements=256,
    memory_ceiling=600000L,
  )
  try
    @luna_cuda_precision.decode_source(tile, symbol="invalid_expansion")
  catch {
    UnsupportedPlacement => ()
    _ => fail("wrong placement failure")
  } noraise {
    _ => fail("whole tensor output escaped tile scratch budget")
  }
}
