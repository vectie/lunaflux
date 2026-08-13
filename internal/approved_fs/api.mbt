///|
pub fn open_root(path : Bytes, status : Ref[Int]) -> ApprovedHandle {
  raw_open_root(path, status)
}

///|
pub fn open_file(
  root : ApprovedHandle,
  locator : Bytes,
  status : Ref[Int],
) -> ApprovedHandle {
  raw_open_file(root, locator, status)
}

///|
pub fn close(handle : ApprovedHandle) -> Int {
  raw_close(handle)
}

///|
pub fn is_closed(handle : ApprovedHandle) -> Bool {
  raw_is_closed(handle)
}

///|
pub fn read_exact_at(
  file : ApprovedHandle,
  destination : FixedArray[Byte],
  destination_offset : Int,
  byte_count : Int,
  position : Int64,
) -> Int {
  raw_read_exact_at(file, destination, destination_offset, byte_count, position)
}

///|
pub fn stamp(file : ApprovedHandle, output : FixedArray[Int64]) -> Int {
  raw_stamp(file, output)
}
