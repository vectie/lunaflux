///|
/// Allocates bounded inline block-table identity storage during startup.
pub fn BlockTableIdStorage::new(
  capacity : Int,
) -> BlockTableIdStorage raise BlockTableError {
  if capacity <= 0 || capacity > ABSOLUTE_MAX_BLOCK_TABLE_ID_STORAGE_CAPACITY {
    raise InvalidLimit(StorageCapacity)
  }
  {
    tables: FixedArray::make(capacity, invalid_block_table_id()),
    occupied: FixedArray::make(capacity, false),
  }
}

///|
pub fn BlockTableIdStorage::capacity(self : BlockTableIdStorage) -> Int {
  self.tables.length()
}

///|
fn BlockTableIdStorage::check_index(
  self : BlockTableIdStorage,
  index : Int,
) -> Unit raise BlockTableError {
  if index < 0 || index >= self.tables.length() {
    raise StorageIndexOutOfBounds
  }
}

///|
pub fn BlockTableIdStorage::is_occupied(
  self : BlockTableIdStorage,
  index : Int,
) -> Bool raise BlockTableError {
  self.check_index(index)
  self.occupied[index]
}

///|
/// Copies one canonical inline identity into an occupied cell.
///
/// The storage does not own the table and cannot validate arena residency.
/// The caller retains that lifecycle and validation duty.
pub fn BlockTableIdStorage::write(
  self : BlockTableIdStorage,
  index : Int,
  table : BlockTableId,
) -> Unit raise BlockTableError {
  self.check_index(index)
  self.tables[index] = table
  self.occupied[index] = true
}

///|
/// Returns a copied canonical identity only from an occupied cell.
pub fn BlockTableIdStorage::read(
  self : BlockTableIdStorage,
  index : Int,
) -> BlockTableId raise BlockTableError {
  self.check_index(index)
  if !self.occupied[index] {
    raise StorageCellEmpty
  }
  self.tables[index]
}

///|
/// Clears occupancy without exposing the package-private sentinel.
pub fn BlockTableIdStorage::clear(
  self : BlockTableIdStorage,
  index : Int,
) -> Unit raise BlockTableError {
  self.check_index(index)
  self.tables[index] = invalid_block_table_id()
  self.occupied[index] = false
}
