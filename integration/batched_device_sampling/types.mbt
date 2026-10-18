///|
priv enum State {
  Ready
  Prepared
  AwaitingInput
  Executing
  Downloading
  Retired
  Failed
  Closed
}

///|
/// Owns a captured device sampler and bounded parameter/result mailboxes.
/// Logits stay on the GPU. Close this owner before its borrowed model storage
/// and AOT module. The caller retires model execution before calling begin.
pub struct BatchedDeviceSampler {
  priv program : @physical.FullVocabularySamplingProgram
  priv control : @device.Allocation
  priv workspace : @device.Allocation
  priv stages : @device.Allocation
  priv results : @device.Allocation
  priv upload : @device.TransferPool
  priv download : @device.TransferPool
  priv stream : @device.Stream
  priv functions : Array[@device.Function]
  priv executor : @device.OrderedKernelExecutor
  priv parameters : FixedArray[Byte]
  priv output : FixedArray[Byte]
  priv mut count : Int
  priv mut state : State
}

///|
fn put_u64(bytes : FixedArray[Byte], offset : Int, value : UInt64) -> Unit {
  for i in 0..<8 {
    bytes[offset + i] = ((value >> (8 * i)) & 255UL).to_byte()
  }
}

///|
fn put_i32(bytes : FixedArray[Byte], offset : Int, value : Int) -> Unit {
  for i in 0..<4 {
    bytes[offset + i] = ((value.reinterpret_as_uint() >> (8 * i)) & 255U).to_byte()
  }
}

///|
fn get_i32(bytes : FixedArray[Byte], offset : Int) -> Int {
  for i in 0..<4; bits = 0U {
    continue bits | (bytes[offset + i].to_uint() << (8 * i))
  } nobreak {
    bits.reinterpret_as_int()
  }
}

///|
fn argument(
  allocation : @device.Allocation,
  offset? : Int64 = 0L,
  bytes? : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    allocation,
    offset~,
    byte_count=bytes.unwrap_or(allocation.byte_count()),
    alignment=8L,
  )
}
