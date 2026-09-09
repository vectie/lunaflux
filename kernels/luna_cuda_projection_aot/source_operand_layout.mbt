///|
/// Physical interpretation of an immutable logical BF16 operand tile. Padding
/// changes addresses only: the reduction order and the logical tensor do not
/// change. This CUDA-local layout keeps each WMMA row and asynchronous vector
/// copy aligned while rotating successive rows across shared-memory banks.
priv struct CudaProjectionOperandLayout {
  row_stride : Int
}

///|
fn cuda_projection_operand_layout(
  logical_width : Int,
  padded? : Bool = true,
) -> CudaProjectionOperandLayout {
  // WMMA's BF16 operand rows require a multiple-of-eight element stride;
  // sixteen elements also preserve the 32-byte alignment of every tile base.
  // An extra sixteen BF16 elements breaks the power-of-two row-bank alias.
  { row_stride: logical_width + (if padded { 16 } else { 0 }) }
}

///|
fn CudaProjectionOperandLayout::row_stride(
  self : CudaProjectionOperandLayout,
) -> Int {
  self.row_stride
}

///|
fn CudaProjectionOperandLayout::offset(
  self : CudaProjectionOperandLayout,
  row : Int,
  column : Int,
) -> Int {
  row * self.row_stride + column
}

///|
fn CudaProjectionOperandLayout::storage_elements(
  self : CudaProjectionOperandLayout,
  rows : Int,
) -> Int {
  rows * self.row_stride
}
