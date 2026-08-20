///|
fn destination_available(
  destination : FixedArray[Byte],
  offset : Int,
  required : Int,
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
  source_offset : Int,
  destination : FixedArray[Byte],
  destination_offset : Int,
  count : Int,
) -> Unit {
  for index in 0..<count {
    destination[destination_offset + index] = source[source_offset + index]
  }
}
