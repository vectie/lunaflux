///|
/// Diagnostic output observation after the entire prepared decoder retires.
/// Not a numerical oracle and not inserted into any production kernel queue.
fn waveform_statistics(bytes : Bytes) -> (Int, Float) {
  for i in 0..<(bytes.length() / 4); nonfinite = 0, maximum = 0.0F {
    let offset = i * 4
    let bits = bytes[offset].to_uint() |
      (bytes[offset + 1].to_uint() << 8) |
      (bytes[offset + 2].to_uint() << 16) |
      (bytes[offset + 3].to_uint() << 24)
    let value = Float::reinterpret_from_uint(bits)
    if value.is_nan() || value.is_inf() {
      continue nonfinite + 1, maximum
    }
    continue nonfinite, Float::max(maximum, value.abs())
  } nobreak {
    (nonfinite, maximum)
  }
}
