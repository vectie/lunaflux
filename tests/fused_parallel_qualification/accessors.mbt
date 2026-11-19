///|
pub fn FusedQkvQualificationFixture::shape(
  self : FusedQkvQualificationFixture,
) -> FusedQualificationShape {
  self.shape
}

///|
pub fn FusedQkvQualificationFixture::input(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.input
}

///|
pub fn FusedQkvQualificationFixture::q_weight(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.q_weight
}

///|
pub fn FusedQkvQualificationFixture::k_weight(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.k_weight
}

///|
pub fn FusedQkvQualificationFixture::v_weight(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.v_weight
}

///|
pub fn FusedQkvQualificationFixture::expected_rotated_qkv(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.rotated_qkv
}

///|
pub fn FusedQkvQualificationFixture::expected_key_cache(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.key_cache
}

///|
pub fn FusedQkvQualificationFixture::expected_value_cache(
  self : FusedQkvQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.value_cache
}

///|
pub fn FusedResidualNormQualificationFixture::shape(
  self : FusedResidualNormQualificationFixture,
) -> FusedQualificationShape {
  self.shape
}

///|
pub fn FusedResidualNormQualificationFixture::residual(
  self : FusedResidualNormQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.residual
}

///|
pub fn FusedResidualNormQualificationFixture::branch(
  self : FusedResidualNormQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.branch
}

///|
pub fn FusedResidualNormQualificationFixture::norm_weight(
  self : FusedResidualNormQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.norm_weight
}

///|
pub fn FusedResidualNormQualificationFixture::expected_residual_output(
  self : FusedResidualNormQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.residual_output
}

///|
pub fn FusedResidualNormQualificationFixture::expected_norm_output(
  self : FusedResidualNormQualificationFixture,
) -> ReadOnlyArray[Int32] {
  self.norm_output
}
