///|
/// Numerical schedule of the pure head-normalization/rotary epilogue. The
/// reduction tree is independent of projection fusion and physical workgroup
/// size: changing either must not silently reassociate this fold.
pub struct AttentionIngressNumerics {
  priv head_dimension : Int
  priv reduction_lanes : Int
  priv norm_epsilon : Float
  priv inverse_frequencies : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub fn ProjectionAttentionIngressEpilogue::numerics(
  self : ProjectionAttentionIngressEpilogue,
  reduction_lanes~ : Int,
) -> AttentionIngressNumerics raise ProjectionTileCompilerError {
  let epsilon = Float::from_double(self.norm_epsilon)
  let theta = Float::from_double(self.rope_theta)
  if self.head_dimension % 2 != 0 ||
    reduction_lanes <= 0 ||
    reduction_lanes > 1024 ||
    (reduction_lanes & (reduction_lanes - 1)) != 0 ||
    epsilon.is_inf() ||
    epsilon <= 0.0F ||
    theta.is_inf() ||
    theta <= 0.0F {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  let frequencies = Array::makei(self.head_dimension / 2, pair => {
    let exponent = Float::from_int(pair * 2) /
      Float::from_int(self.head_dimension)
    1.0F / @math.powf(theta, exponent)
  })
  for frequency in frequencies {
    if frequency.is_nan() || frequency.is_inf() || frequency <= 0.0F {
      raise InvalidProjectionTileCompilation(InvalidProblem)
    }
  }
  {
    head_dimension: self.head_dimension,
    reduction_lanes,
    norm_epsilon: epsilon,
    inverse_frequencies: ReadOnlyArray::from_array(frequencies),
  }
}

///|
pub fn AttentionIngressNumerics::shape(
  self : AttentionIngressNumerics,
) -> (Int, Int) {
  (self.head_dimension, self.reduction_lanes)
}

///|
pub fn AttentionIngressNumerics::epsilon(
  self : AttentionIngressNumerics,
) -> Float {
  self.norm_epsilon
}

///|
/// AOT-invariant basis, shared by fused and separately materialized consumers.
/// Position multiplication and paired rotation retain Float arithmetic; both
/// normalization and rotation have explicit BF16 rounding boundaries.
pub fn AttentionIngressNumerics::rotary_basis(
  self : AttentionIngressNumerics,
) -> ReadOnlyArray[Float] {
  self.inverse_frequencies
}
