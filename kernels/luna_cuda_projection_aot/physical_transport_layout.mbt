///|
/// CUDA transport facts supplied to physical refinement, before emission.
/// This map keeps each logical row contiguous and separates fragment banks.
priv struct CudaProjectionTransportLayout {
  columns : Int
}

///|
fn CudaProjectionTransportLayout::rotations(
  self : CudaProjectionTransportLayout,
) -> Int {
  if self.columns / 8 % 8 == 0 {
    8
  } else if self.columns / 8 % 4 == 0 {
    4
  } else {
    2
  }
}

///|
fn CudaProjectionTransportLayout::checked_plan(
  self : CudaProjectionTransportLayout,
) -> @physical_tile_ir.OperandLayout? {
  @physical_tile_ir.OperandLayout::new(
    columns=self.columns,
    vector=8,
    rotations=self.rotations(),
    row_group=8 / self.rotations(),
  )
}
