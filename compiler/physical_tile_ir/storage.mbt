///|
/// One disjoint immutable operand strip in a reusable storage slot. Producer
/// vector addresses and consumer scalar addresses use this exact layout.
pub struct OperandRegion {
  priv base : Int
  priv rows : Int
  priv layout : OperandLayout
  priv ownership : VectorOwnership
} derive(Eq, Debug)

///|
pub extend OperandRegion with Eq::{equal, not_equal}

///|
pub extend OperandRegion with @debug.Debug::{to_repr}

///|
pub fn OperandRegion::geometry(self : OperandRegion) -> (Int, Int, Int) {
  let (columns, _, _, _) = self.layout.geometry()
  (self.base, self.rows, columns)
}

///|
pub fn OperandRegion::ownership(self : OperandRegion) -> VectorOwnership {
  self.ownership
}

///|
/// Allocation-relative element offset, bounded to this operand's own rows.
pub fn OperandRegion::offset(
  self : OperandRegion,
  row : Int,
  column : Int,
) -> Int? {
  guard row >= 0 && row < self.rows else { return None }
  self.layout.offset(row, column).map(offset => self.base + offset)
}

///|
/// A global-copy owner reads one logical row vector and publishes it at the
/// returned storage offset. Invalid owner/round pairs have no memory effect.
pub fn OperandRegion::copy_at(
  self : OperandRegion,
  owner : Int,
  round : Int,
) -> (Int, Int, Int)? {
  guard self.ownership.vector_at(owner, round) is Some(index) else {
    return None
  }
  let (columns, vector, _, _) = self.layout.geometry()
  let row = index / (columns / vector)
  let column = index % (columns / vector) * vector
  Some((row, column, self.offset(row, column).unwrap()))
}

///|
/// Packed disjoint operand strips, repeated across lifetime-managed slots.
/// Units are scalar elements; backend scalar dtype determines byte size.
pub struct OperandStorage {
  priv regions : ReadOnlyArray[OperandRegion]
  priv layout : OperandLayout
  priv slot_elements : Int
  priv stages : Int
} derive(Eq, Debug)

///|
pub extend OperandStorage with Eq::{equal, not_equal}

///|
pub extend OperandStorage with @debug.Debug::{to_repr}

///|
pub fn OperandStorage::new(
  layout : OperandLayout,
  operand_rows : ArrayView[Int],
  owners~ : Int,
  stages~ : Int,
) -> OperandStorage? {
  guard !operand_rows.is_empty() && owners > 0 && stages > 0 else {
    return None
  }
  let (columns, vector, _, _) = layout.geometry()
  let slot_elements = for rows in operand_rows; total = 0L {
    guard rows > 0 else { return None }
    let next = total + rows.to_int64() * columns.to_int64()
    guard next <= 2147483647L / stages.to_int64() else { return None }
    continue next
  } nobreak {
    total.to_int()
  }
  let regions : Array[OperandRegion] = []
  for rows in operand_rows; base = 0 {
    let elements = rows * columns
    regions.push({
      base,
      rows,
      layout,
      ownership: VectorOwnership::new(vectors=elements / vector, owners~).unwrap(),
    })
    continue base + elements
  }
  Some({
    regions: ReadOnlyArray::from_array(regions),
    layout,
    slot_elements,
    stages,
  })
}

///|
pub fn OperandStorage::region(
  self : OperandStorage,
  index : Int,
) -> OperandRegion? {
  self.regions.get(index)
}

///|
pub fn OperandStorage::layout(self : OperandStorage) -> OperandLayout {
  self.layout
}

///|
pub fn OperandStorage::slot_elements(self : OperandStorage) -> Int {
  self.slot_elements
}

///|
pub fn OperandStorage::allocation_elements(self : OperandStorage) -> Int {
  self.slot_elements * self.stages
}

///|
/// Whether a backend may address these strips as one concatenated matrix
/// without changing their individual row-local permutations.
pub fn OperandStorage::has_concatenated_layout(self : OperandStorage) -> Bool {
  let (columns, _, rotations, row_group) = self.layout.geometry()
  if rotations == 1 {
    return true
  }
  let period = rotations.to_int64() * row_group.to_int64()
  self.regions.all(region => (region.base / columns).to_int64() % period == 0L)
}

///|
pub fn OperandStorage::canonical(self : OperandStorage) -> String {
  let regions = self.regions
    .map(region => {
      let (base, rows, _) = region.geometry()
      let (vectors, owners, _) = region.ownership().geometry()
      "\{base}:\{rows}:\{vectors}:\{owners}"
    })
    .join(",")
  "tile-operand-storage-v1;layout=\{self.layout.canonical()};stages=\{self.stages};regions=\{regions}"
}
