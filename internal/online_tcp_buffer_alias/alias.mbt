///|
#borrow(source)
extern "c" fn raw_retain_bytes_as_fixed_array(
  source : Bytes,
) -> FixedArray[Byte] = "lunaflux_online_tcp_retain_bytes_as_fixed_array"

///|
/// Retains one private dynamic Bytes backing as a mutable byte-array view.
/// Only service/online_tcp may call this representation bridge.
pub fn retain_bytes_as_fixed_array(source : Bytes) -> FixedArray[Byte] {
  raw_retain_bytes_as_fixed_array(source)
}
