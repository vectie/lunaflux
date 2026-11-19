///|
/// Offline coefficient producer. Positions are row-major T/H/W int32 values;
/// host trigonometric primitives are not a bitwise GPU or upstream oracle.
pub fn rotary_coefficients_into(
  plan : @precision.MultiAxisRotaryPrecision,
  positions : ArrayView[Int32],
  cosine : FixedArray[Float],
  sine : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (_, _, rotary, maximum_rows, maximum_positions) = plan.geometry()
  let pairs = rotary / 2
  guard rows > 0 &&
    rows <= maximum_rows &&
    positions.length().to_int64() == rows.to_int64() * 3L &&
    cosine.length().to_int64() == rows.to_int64() * pairs.to_int64() &&
    sine.length() == cosine.length() &&
    positions.all(position => position >= 0 && position < maximum_positions) else {
    fail("invalid rotary coefficient buffers")
  }
  for row in 0..<rows {
    for pair in 0..<pairs {
      let angle = Float::from_int(positions[row * 3 + plan.axes()[pair]]) *
        plan.inverse_frequencies()[pair]
      cosine[row * pairs + pair] = round_bf16(@math.cosf(angle))
      sine[row * pairs + pair] = round_bf16(@math.sinf(angle))
    }
  }
}

///|
/// Packed heads with BF16-valued coefficients/inputs. Split prefix halves
/// rotate; the tail is copied verbatim. No run allocation; output may alias input.
pub fn multi_axis_rotary_into(
  plan : @precision.MultiAxisRotaryPrecision,
  input : ArrayView[Float],
  cosine : ArrayView[Float],
  sine : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (heads, width, rotary, maximum_rows, _) = plan.geometry()
  let pairs = rotary / 2
  guard rows > 0 &&
    rows <= maximum_rows &&
    input.length().to_int64() ==
    rows.to_int64() * heads.to_int64() * width.to_int64() &&
    output.length() == input.length() &&
    cosine.length().to_int64() == rows.to_int64() * pairs.to_int64() &&
    sine.length() == cosine.length() else {
    fail("invalid rotary buffers")
  }
  for row in 0..<rows {
    for head in 0..<heads {
      let base = (row * heads + head) * width
      for pair in 0..<pairs {
        let left = round_bf16(input[base + pair])
        let right = round_bf16(input[base + pair + pairs])
        let c = round_bf16(cosine[row * pairs + pair])
        let s = round_bf16(sine[row * pairs + pair])
        output[base + pair] = round_bf16(
          round_bf16(left * c) - round_bf16(right * s),
        )
        output[base + pair + pairs] = round_bf16(
          round_bf16(right * c) + round_bf16(left * s),
        )
      }
      for column in rotary..<width {
        output[base + column] = input[base + column]
      }
    }
  }
}
