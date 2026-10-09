///|
/// Shared CUDA frequency lowering for query/KV rotation and inverse output
/// rotation. Caller declares `pair`; the result declares `frequency` in F32.
pub fn rotary_frequency_source(
  frequency : @precision.RotaryFrequency,
  rotary_width : Int,
) -> String raise {
  guard rotary_width > 0 && rotary_width % 2 == 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  match frequency {
    Base(theta~) =>
      "const float frequency = __fdiv_rn(1.0f, powf(" +
      @literal.canonical_cuda_float_literal(theta) +
      ", __fdiv_rn((float)(pair * 2), \{rotary_width}.0f)));"
    Yarn(theta~, original_positions~, factor~, beta_fast~, beta_slow~) => {
      let theta_f = @literal.canonical_cuda_float_literal(theta)
      let factor_f = @literal.canonical_cuda_float_literal(factor)
      (
        $|const double fast = ((double)\{rotary_width} * log((double)\{original_positions} / ((double)\{beta_fast} * 6.2831853071795864769))) / (2.0 * log((double)\{theta}));
        $|const double slow = ((double)\{rotary_width} * log((double)\{original_positions} / ((double)\{beta_slow} * 6.2831853071795864769))) / (2.0 * log((double)\{theta}));
        $|const float low = (float)fmax(0.0, floor(fast)); const float high = (float)fmin(\{rotary_width - 1}.0, ceil(slow));
        $|const float span = __fadd_rn(high, __fadd_rn(-low, high == low ? 0.001f : 0.0f));
        $|const float ramp = fminf(1.0f, fmaxf(0.0f, __fdiv_rn(__fadd_rn((float)pair, -low), span)));
        $|const float smooth = __fadd_rn(1.0f, -ramp);
        $|const float base_frequency = __fdiv_rn(1.0f, powf(\{theta_f}, __fdiv_rn((float)(pair * 2), \{rotary_width}.0f)));
        $|const float frequency = __fadd_rn(__fmul_rn(__fdiv_rn(base_frequency, \{factor_f}), __fadd_rn(1.0f, -smooth)), __fmul_rn(base_frequency, smooth));
      )
    }
  }
}
