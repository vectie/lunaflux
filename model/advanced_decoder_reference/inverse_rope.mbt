///|
fn inverse_rope_finite_positive(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf() && value > 0.0
}

///|
/// Applies official adjacent-pair inverse YaRN rotation to the RoPE suffix of
/// one BF16-boundary attention row. The non-RoPE prefix is copied unchanged.
/// This bounded host oracle performs F32 arithmetic but does not emulate the
/// final BF16 rounding performed by the CUDA boundary.
pub fn attention_output_inverse_rotary_v1(
  attention_output : ArrayView[Float],
  position~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  rope_dimension~ : Int,
  maximum_positions~ : Int,
  original_positions~ : Int,
  compressed_theta~ : Double,
  scaling_factor~ : Double,
  beta_fast~ : Double,
  beta_slow~ : Double,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  if head_count <= 0 ||
    head_dimension <= 0 ||
    rope_dimension <= 0 ||
    rope_dimension % 2 != 0 ||
    rope_dimension > head_dimension ||
    maximum_positions <= 0 ||
    original_positions <= 0 ||
    original_positions > maximum_positions ||
    position < 0 ||
    position >= maximum_positions ||
    !inverse_rope_finite_positive(compressed_theta) ||
    !inverse_rope_finite_positive(scaling_factor) ||
    !inverse_rope_finite_positive(beta_fast) ||
    !inverse_rope_finite_positive(beta_slow) ||
    head_count > limits.max_width() ||
    attention_output.length() != head_count * head_dimension ||
    attention_output.length() > limits.max_values() {
    raise InvalidDimension
  }
  validate_values(attention_output, limits)
  let correction_fast = rope_dimension.to_double() *
    @math.ln(
      original_positions.to_double() / (beta_fast * 6.2831853071795864769),
    ) /
    (2.0 * @math.ln(compressed_theta))
  let correction_slow = rope_dimension.to_double() *
    @math.ln(
      original_positions.to_double() / (beta_slow * 6.2831853071795864769),
    ) /
    (2.0 * @math.ln(compressed_theta))
  let low_value = @math.floor(correction_fast)
  let low = Float::from_double(if low_value < 0.0 { 0.0 } else { low_value })
  let high_value = @math.ceil(correction_slow)
  let high = Float::from_double(
    if high_value > (rope_dimension - 1).to_double() {
      (rope_dimension - 1).to_double()
    } else {
      high_value
    },
  )
  let theta = Float::from_double(compressed_theta)
  let factor = Float::from_double(scaling_factor)
  let position_f = Float::from_int(position)
  let prefix = head_dimension - rope_dimension
  let output = attention_output.to_owned()
  for head in 0..<head_count {
    let base = head * head_dimension + prefix
    for pair in 0..<(rope_dimension / 2) {
      let denominator = if high == low { 0.001F } else { high - low }
      let raw_ramp = (Float::from_int(pair) - low) / denominator
      let ramp = if raw_ramp < 0.0F {
        0.0F
      } else if raw_ramp > 1.0F {
        1.0F
      } else {
        raw_ramp
      }
      let smooth = 1.0F - ramp
      let exponent = Float::from_int(pair * 2) / Float::from_int(rope_dimension)
      let base_frequency = 1.0F / @math.powf(theta, exponent)
      let frequency = base_frequency / factor * (1.0F - smooth) +
        base_frequency * smooth
      let angle = position_f * frequency
      let cosine = @math.cosf(angle)
      let sine = @math.sinf(angle)
      let index = base + pair * 2
      let left = attention_output[index]
      let right = attention_output[index + 1]
      output[index] = left * cosine + right * sine
      output[index + 1] = -left * sine + right * cosine
    }
  }
  ReadOnlyArray::from_array(output)
}
