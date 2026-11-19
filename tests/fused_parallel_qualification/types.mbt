///|
pub(all) suberror FusedQualificationError {
  InvalidDeclaredShape
  InvalidRefereeInput
  RefereeMismatch(index~ : Int32)
} derive(Debug, Eq)

///|
/// Exact integer-PPB summary produced directly from BF16 referee cells.
///
/// The type is observation-only. Its fields are private so callers cannot
/// manufacture a passing numerical result without running a referee.
pub struct FusedQualificationComparisonSummary {
  priv maximum_absolute_error_ppb : Int64
  priv maximum_relative_error_ppb : Int64
  priv maximum_absolute_error : Float
}

///|
pub fn FusedQualificationComparisonSummary::maximum_absolute_error_ppb(
  self : FusedQualificationComparisonSummary,
) -> Int64 {
  self.maximum_absolute_error_ppb
}

///|
pub fn FusedQualificationComparisonSummary::maximum_relative_error_ppb(
  self : FusedQualificationComparisonSummary,
) -> Int64 {
  self.maximum_relative_error_ppb
}

///|
pub fn FusedQualificationComparisonSummary::maximum_absolute_error(
  self : FusedQualificationComparisonSummary,
) -> Float {
  self.maximum_absolute_error
}

///|
/// One immutable, source-declared qualification shape. It is test evidence,
/// not a launch contract or runtime shape class.
pub struct FusedQualificationShape {
  priv ordinal : Int32
  priv name : String
  priv token_count : Int32
  priv positions : ReadOnlyArray[Int32]
  priv page_table : ReadOnlyArray[Int32]
}

///|
pub fn FusedQualificationShape::ordinal(
  self : FusedQualificationShape,
) -> Int32 {
  self.ordinal
}

///|
pub fn FusedQualificationShape::name(self : FusedQualificationShape) -> String {
  self.name
}

///|
pub fn FusedQualificationShape::token_count(
  self : FusedQualificationShape,
) -> Int32 {
  self.token_count
}

///|
pub fn FusedQualificationShape::positions(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int32] {
  self.positions
}

///|
pub fn FusedQualificationShape::page_table(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int32] {
  self.page_table
}

///|
pub struct FusedQkvQualificationFixture {
  priv shape : FusedQualificationShape
  priv input : ReadOnlyArray[Int32]
  priv q_weight : ReadOnlyArray[Int32]
  priv k_weight : ReadOnlyArray[Int32]
  priv v_weight : ReadOnlyArray[Int32]
  priv rotated_qkv : ReadOnlyArray[Int32]
  priv key_cache : ReadOnlyArray[Int32]
  priv value_cache : ReadOnlyArray[Int32]
}

///|
pub struct FusedResidualNormQualificationFixture {
  priv shape : FusedQualificationShape
  priv residual : ReadOnlyArray[Int32]
  priv branch : ReadOnlyArray[Int32]
  priv norm_weight : ReadOnlyArray[Int32]
  priv residual_output : ReadOnlyArray[Int32]
  priv norm_output : ReadOnlyArray[Int32]
}
