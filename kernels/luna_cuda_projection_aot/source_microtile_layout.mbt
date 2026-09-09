///|
/// Immutable CUDA operand permutation. Admitted projection folds have rows
/// and columns divisible by the 16-wide matrix tile. Eight-wide subtiles preserve
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
  (row / 8 * (self.columns / 8) + column / 8) * 64 + row % 8 * 8 + column % 8
}

///|
fn CudaProjectionMicrotileLayout::address_function(
  self : CudaProjectionMicrotileLayout,
) -> String {
  (
    $|auto operand_index = [](int row, int column) {
    $|      return ((row / 8) * \{self.columns / 8} + column / 8) * 64 +
    $|          (row % 8) * 8 + column % 8;
    $|    };
  )
}
