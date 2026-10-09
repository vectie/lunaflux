///|
/// Compact parameter-buffer layout shared by checkpoint transfer and lowering.
/// Scale planes use natural scalar alignment; payload packing stays unchanged.
pub struct PackedBufferLayout {
  priv scales : Int64?
  priv global : Int64?
  priv bytes : Int64
}

///|
pub fn packed_buffer_layout(
  rep : @format.Representation,
) -> PackedBufferLayout raise PrecisionPlanError {
  if rep.payload_bytes() > 9223372036854775807L - rep.scale_bytes() - 7L {
    raise ArithmeticOverflow
  }
  let scale_start = if rep.scale_element() == F32 {
    (rep.payload_bytes() + 3L) / 4L * 4L
  } else {
    rep.payload_bytes()
  }
  let end = scale_start + rep.scale_bytes()
  let global_start = (end + 3L) / 4L * 4L
  {
    scales: if rep.scale_bytes() == 0L {
      None
    } else {
      Some(scale_start)
    },
    global: if rep.has_global_scale() {
      Some(global_start)
    } else {
      None
    },
    bytes: if rep.has_global_scale() {
      global_start + 4L
    } else {
      end
    },
  }
}

///|
pub fn PackedBufferLayout::scales_offset(self : Self) -> Int64? {
  self.scales
}

///|
pub fn PackedBufferLayout::global_scale_offset(self : Self) -> Int64? {
  self.global
}

///|
pub fn PackedBufferLayout::buffer_bytes(self : Self) -> Int64 {
  self.bytes
}
