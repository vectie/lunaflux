///|
/// Immutable CUDA operand permutation. Admitted projection folds have rows
/// and columns divisible by the 16-wide WMMA tile. Compact tiles preserve
/// capacity, ordered reduction and aligned eight-element asynchronous copies.
priv struct CudaProjectionMicrotileLayout {
  columns : Int
}

///|
fn CudaProjectionMicrotileLayout::offset(
  self : CudaProjectionMicrotileLayout,
  row : Int,
  column : Int,
) -> Int {
  (row / 16 * (self.columns / 16) + column / 16) * 256 +
  row % 16 * 16 +
  column % 16
}

///|
fn CudaProjectionMicrotileLayout::address_function(
  self : CudaProjectionMicrotileLayout,
) -> String {
  (
    $|auto operand_index = [](int row, int column) {
    $|      return ((row / 16) * \{self.columns / 16} + column / 16) * 256 +
    $|          (row % 16) * 16 + column % 16;
    $|    };
  )
}
