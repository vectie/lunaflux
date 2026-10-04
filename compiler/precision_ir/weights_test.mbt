///|
test "weight schema accounts for all tensors and explicit scratch lifetimes" {
  let accuracy = @precision_ir.Accuracy::new(absolute=0.1, relative=0.1)
  let operand = @precision_ir.plan_operand(
    role=Weight,
    representation=@format.Representation::new(format=Int4, rows=2, columns=7),
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy~,
    tile_elements=32,
    memory_ceiling=1000L,
  )
  let serial = @precision_ir.WeightSchema::new(
    [operand, operand],
    alignment_bytes=64L,
    scratch_lifetime=Sequential,
    device_memory_ceiling=144L,
  )
  assert_eq(serial.memory(), (80L, 64L))
  assert_eq(serial.bindings()[1].offset(), 64L)
  assert_eq(serial.bindings()[1].ordinal(), 1)
  let parallel = @precision_ir.WeightSchema::new(
    [operand, operand],
    alignment_bytes=64L,
    scratch_lifetime=Concurrent,
    device_memory_ceiling=208L,
  )
  assert_eq(parallel.memory(), (80L, 128L))
  assert_true(serial.canonical() != parallel.canonical())
  try
    @precision_ir.WeightSchema::new(
      [operand, operand],
      alignment_bytes=64L,
      scratch_lifetime=Concurrent,
      device_memory_ceiling=207L,
    )
  catch {
    MemoryBudgetExceeded => ()
    _ => fail("wrong memory failure")
  } noraise {
    _ => fail("model budget was replaced by per-weight budget")
  }
}

///|
test "startup BF16 schema does not pretend compact device memory" {
  let operand = @precision_ir.plan_operand(
    role=Weight,
    representation=@format.Representation::new(
      format=NvFp4,
      rows=32,
      columns=128,
    ),
    placement=StartupExpandBf16,
    supported_placements=[StartupExpandBf16],
    accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
    memory_ceiling=8192L,
  )
  let schema = @precision_ir.WeightSchema::new(
    [operand, operand],
    alignment_bytes=256L,
    scratch_lifetime=Sequential,
    device_memory_ceiling=16384L,
  )
  assert_eq(schema.memory(), (16384L, 0L))
}
