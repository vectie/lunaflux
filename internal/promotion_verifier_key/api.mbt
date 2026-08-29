///|
pub fn open_fixed() -> PromotionVerifierKeyHandle {
  { native: raw_open_fixed() }
}

///|
pub fn open_status(handle : PromotionVerifierKeyHandle) -> Int {
  raw_open_status(handle.native)
}

///|
pub fn try_read(
  handle : PromotionVerifierKeyHandle,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int {
  raw_try_read(handle.native, destination, offset, length)
}

///|
pub fn peek(handle : PromotionVerifierKeyHandle) -> Int {
  raw_peek(handle.native)
}

///|
pub fn close(handle : PromotionVerifierKeyHandle) -> Int {
  raw_close(handle.native)
}

///|
pub fn is_closed(handle : PromotionVerifierKeyHandle) -> Bool {
  raw_is_closed(handle.native)
}
