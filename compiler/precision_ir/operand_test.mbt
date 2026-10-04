///|
test "placement distinguishes compact tile decode from startup expansion" {
  let rep = @format.Representation::new(format=NvFp4, rows=128, columns=128)
  let accuracy = @precision_ir.Accuracy::new(absolute=0.1, relative=0.1)
  let tile = @precision_ir.plan_operand(
    role=Weight,
    representation=rep,
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy~,
    memory_ceiling=100000L,
    tile_elements=256,
  )
  assert_eq(tile.memory(), (9220L, 512L))
  let expand = @precision_ir.plan_operand(
    role=Weight,
    representation=rep,
    placement=StartupExpandBf16,
    supported_placements=[StartupExpandBf16],
    accuracy~,
    memory_ceiling=32768L,
  )
  assert_eq(expand.memory(), (32768L, 0L))
  assert_eq(expand.effect(), @precision_ir.NumericEffect::StartupWrite)
  assert_true(tile.canonical() != expand.canonical())
}

///|
test "cache quantization is an explicit commit and cannot become weight initialization" {
  let rep = @format.Representation::new(format=MxFp8E4M3, rows=4, columns=128)
  let accuracy = @precision_ir.Accuracy::new(absolute=0.1, relative=0.1)
  let kv = @precision_ir.plan_operand(
    role=KeyCache,
    representation=rep,
    placement=QuantizeOnCacheWrite,
    supported_placements=[QuantizeOnCacheWrite],
    accuracy~,
    memory_ceiling=10000L,
  )
  assert_eq(kv.effect(), @precision_ir.NumericEffect::CacheCommit)
  try
    @precision_ir.plan_operand(
      role=KeyCache,
      representation=rep,
      placement=StartupExpandBf16,
      supported_placements=[StartupExpandBf16],
      accuracy~,
      memory_ceiling=10000L,
    )
  catch {
    IllegalPlacement => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("cache ownership lost")
  }
  try
    @precision_ir.plan_operand(
      role=Weight,
      representation=rep,
      placement=NativeOperand,
      supported_placements=[DecodeInOperandTile],
      accuracy~,
      memory_ceiling=10000L,
    )
  catch {
    NoImplementation => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("capability fabricated")
  }
}
