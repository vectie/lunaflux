///|
/// Offline capture of all retired physical vocabulary rows. The explicit egress
/// diagnostic mode prepares this transfer before serving, then reads only once
/// after control EOF. It never changes the execution graph or token steps.
async fn capture_retired_logits(
  pool : @device.TransferPool,
  bytes : Int,
  destination : String,
  filename? : String = "logits.f32",
  create_directory? : Bool = true,
) -> Unit {
  pool.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=bytes.to_int64(),
    to_device=false,
  )
  pool.record(0)
  while !pool.poll(0) {
    @async.pause()
  }
  let payload = FixedArray::make(bytes, b'\x00')
  pool.read_host(payload, host_offset=0L, destination_offset=0, bytes~)
  pool.reset(0)
  if create_directory {
    @fs.mkdir(destination)
  }
  @fs.write_file(
    destination + "/" + filename,
    Bytes::from_array(payload[:]),
    create_mode=CreateNew,
  )
  println(
    "retired_tensor_capture file=\{filename} bytes=\{bytes} scope=allocated_physical_rows",
  )
}
