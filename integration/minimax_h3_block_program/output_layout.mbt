///|
struct OutputLayout {
  parameters : Int64
  selected : Int64
  bytes : Int64
}

///|
fn output_layout(
  rows : Int,
  width : Int,
  times : Int,
  video : Int,
  audio : Int,
  budget : Int64,
) -> OutputLayout raise @device.DeviceError {
  guard rows > 0 &&
    width > 0 &&
    times > 0 &&
    video > 0 &&
    audio > 0 &&
    video <= rows &&
    audio <= rows - video &&
    rows.to_int64() <= 2147483647L / width.to_int64() &&
    times.to_int64() <= 2147483647L / 2L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  fn align(bytes : Int64) -> Int64 {
    bytes + (256L - bytes % 256L) % 256L
  }
  let parameters = align(rows.to_int64() * width.to_int64() * 2L)
  let selected = parameters + align(times.to_int64() * width.to_int64() * 4L)
  // Selected media hidden states have disjoint lifetimes. Video projection
  // finishes before audio modulation overwrites this F32 region.
  let bytes = selected +
    Int::max(video, audio).to_int64() * width.to_int64() * 4L
  guard budget > 0L && bytes <= budget else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  { parameters, selected, bytes, }
}
