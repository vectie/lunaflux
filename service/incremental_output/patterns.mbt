///|
fn destination_available(
  destination : FixedArray[Byte],
  offset : Int32,
  required : Int32,
) -> Unit raise IncrementalOutputError {
  if offset < 0 ||
    offset > destination.length() ||
    required > destination.length() - offset {
    raise invalid(Destination, TooSmall)
  }
}

///|
fn copy_range(
  source : FixedArray[Byte],
  source_offset : Int32,
  destination : FixedArray[Byte],
  destination_offset : Int32,
  count : Int32,
) -> Unit {
  for index in 0..<count {
    destination[destination_offset + index] = source[source_offset + index]
  }
}
