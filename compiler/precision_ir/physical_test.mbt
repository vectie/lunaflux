///|
test "precision refinement shares physical producer consumer layout and ordered rounding" {
  let representation = @format.Representation::new(
    format=NvFp4,
    rows=128,
    columns=128,
  )
  let plan = @precision_ir.plan_operand(
    role=Weight,
    representation~,
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy=@precision_ir.Accuracy::new(absolute=0.2, relative=0.1),
    memory_ceiling=100000L,
    tile_elements=256,
  )
  let layout = @physical.OperandLayout::new(
    columns=16,
    vector=8,
    rotations=2,
    row_group=1,
  ).unwrap()
  let region = @physical.OperandStorage::new(layout, [16], owners=32, stages=1)
    .unwrap()
    .region(0)
    .unwrap()
  let refined = plan.refine_region(region)
  assert_eq(
    refined.conversions(),
    ReadOnlyArray::from_array([
      @precision_ir.Conversion::DecodeElement,
      ApplyLocalScaleF32,
      ApplyGlobalScaleF32,
      RoundBf16,
    ]),
  )
  assert_eq(refined.region().offset(1, 0), Some(24))
  assert_eq(refined.canonical(), plan.refine_region(region).canonical())
}
