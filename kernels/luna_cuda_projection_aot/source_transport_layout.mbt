///|
/// Private lowering of an immutable operand map. Unlike compact microtiles,
/// this permutation keeps each logical row in one contiguous physical row,
/// while separating the banks read by packed matrix-fragment lanes.
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
fn CudaProjectionTransportLayout::offset(
  self : CudaProjectionTransportLayout,
  row : Int,
  column : Int,
) -> Int {
  let rotations = self.rotations()
  row * self.columns + (column ^ (row / (8 / rotations) % rotations * 8))
}

///|
fn CudaProjectionTransportLayout::address_function(
  self : CudaProjectionTransportLayout,
) -> String {
  let rotations = self.rotations()
  // The permutation repeats every eight rows; derive its physical stride
  // from the same map that is checked by the producer/consumer tests.
  let stride = self.offset(8, 0) / 8
  let rotation = if rotations == 8 {
    "r & 7"
  } else {
    "(r / \{8 / rotations}) & \{rotations - 1}"
  }
  (
    $|auto operand_index = [](int row, int column) {
    $|      const unsigned r = (unsigned)row, c = (unsigned)column;
    $|      return r * \{stride} + (c ^ ((\{rotation}) * 8));
    $|    };
    $|    auto result_index = [](int row, int column) {
    $|      const unsigned r = (unsigned)row & 7, c = (unsigned)column;
    $|      const unsigned bank = (r * 4) ^ ((r & 1) * 8) ^ (c >> 1) ^ ((c & 1) * 16);
    $|      return ((unsigned)row >> 3) * 128 + ((c >> 3) + (c & 1) * 2) * 32 + bank;
    $|    };
  )
}
