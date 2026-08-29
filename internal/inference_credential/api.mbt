///|
pub fn open_fixed() -> InferenceCredentialHandle {
  { native: raw_open_fixed() }
}

///|
pub fn open_status(handle : InferenceCredentialHandle) -> Int {
  raw_open_status(handle.native)
}

///|
pub fn try_read(
  handle : InferenceCredentialHandle,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int {
  raw_try_read(handle.native, destination, offset, length)
}

///|
pub fn peek(handle : InferenceCredentialHandle) -> Int {
  raw_peek(handle.native)
}

///|
pub fn close(handle : InferenceCredentialHandle) -> Int {
  raw_close(handle.native)
}

///|
pub fn is_closed(handle : InferenceCredentialHandle) -> Bool {
  raw_is_closed(handle.native)
}
