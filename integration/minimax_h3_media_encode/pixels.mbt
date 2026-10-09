///|
/// Bounded ingress conversion of already resized/cropped RGB24 [T,H,W,3] to
/// ImageNet-normalized F32 [3,T,H,W]. Media decoding/resizing happens once
/// upstream so the visual tokenizer and VAE share exactly the same pixels.
pub fn rgb_chunk(
  rgb : BytesView,
  start_element~ : Int,
  count~ : Int,
  frames~ : Int,
  height~ : Int,
  width~ : Int,
) -> Bytes raise @plan.PlanError {
  guard frames > 0 &&
    height > 0 &&
    width > 0 &&
    count > 0 &&
    count <= 536870911 &&
    start_element >= 0 else {
    raise @plan.PlanError::InvalidGeometry
  }
  let plane = frames.to_int64() * height.to_int64()
  guard plane <= 2147483647L / width.to_int64() / 3L else {
    raise @plan.PlanError::InvalidGeometry
  }
  let plane = (plane * width.to_int64()).to_int()
  guard rgb.length() == plane * 3 &&
    start_element <= rgb.length() &&
    count <= rgb.length() - start_element else {
    raise @plan.PlanError::InvalidGeometry
  }
  let mean : Array[Float] = [0.485, 0.456, 0.406]
  let std : Array[Float] = [0.229, 0.224, 0.225]
  let bytes : Array[Byte] = Array::make(count * 4, 0)
  for i in 0..<count {
    let flat = start_element + i
    let channel = flat / plane
    let pixel = rgb[flat % plane * 3 + channel].to_int().to_float()
    let normalized = (pixel / 255.0 - mean[channel]) / std[channel]
    let bits = normalized.reinterpret_as_uint()
    for byte in 0..<4 {
      bytes[i * 4 + byte] = ((bits >> (byte * 8)) & 255U).to_byte()
    }
  }
  Bytes::from_array(bytes)
}

///|
/// Upload prepared RGB in bounded conversion chunks without allocating a
/// complete host F32 tensor. Failed upload is retried before binding execution.
pub fn upload_rgb(
  context : @device.Context,
  destination : @device.Allocation,
  source : VideoSource,
  rgb : BytesView,
  destination_offset~ : Int64,
  chunk_elements~ : Int,
) -> Unit raise {
  let count = source.plan.graph.values[0].elements
  guard chunk_elements > 0 &&
    chunk_elements <= 536870911 &&
    rgb.length().to_int64() == count else {
    raise @plan.PlanError::InvalidGeometry
  }
  context.validate_allocation_region(
    destination,
    offset=destination_offset,
    byte_count=count * 4L,
    alignment=4L,
  )
  for start = 0; start < rgb.length(); {
    let size = Int::min(chunk_elements, rgb.length() - start)
    let bytes = rgb_chunk(
      rgb,
      start_element=start,
      count=size,
      frames=source.plan.frames,
      height=source.plan.height,
      width=source.plan.width,
    )
    destination.copy_from_host(
      bytes,
      0L,
      destination_offset + start.to_int64() * 4L,
      size.to_int64() * 4L,
    )
    continue start + size
  }
}
