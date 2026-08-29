///|
pub fn open_fixed() -> InheritedDrainHandle {
  { native: raw_open_fixed() }
}

///|
pub fn open_status(handle : InheritedDrainHandle) -> Int {
  raw_open_status(handle.native)
}

///|
pub fn try_read(
  handle : InheritedDrainHandle,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int {
  raw_try_read(handle.native, destination, offset, length)
}

///|
pub fn peek(handle : InheritedDrainHandle) -> Int {
  raw_peek(handle.native)
}

///|
pub fn try_write(
  handle : InheritedDrainHandle,
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int {
  raw_try_write(handle.native, source, offset, length)
}

///|
pub fn close(handle : InheritedDrainHandle) -> Int {
  raw_close(handle.native)
}

///|
pub fn is_closed(handle : InheritedDrainHandle) -> Bool {
  raw_is_closed(handle.native)
}
