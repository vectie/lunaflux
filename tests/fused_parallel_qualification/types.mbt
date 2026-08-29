///|
pub(all) suberror FusedQualificationError {
  InvalidDeclaredShape
  InvalidRefereeInput
  RefereeMismatch(index~ : Int)
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
  priv ordinal : Int
  priv name : String
  priv token_count : Int
  priv positions : ReadOnlyArray[Int]
  priv page_table : ReadOnlyArray[Int]
}

///|
pub fn FusedQualificationShape::ordinal(self : FusedQualificationShape) -> Int {
  self.ordinal
}

///|
pub fn FusedQualificationShape::name(self : FusedQualificationShape) -> String {
  self.name
}

///|
pub fn FusedQualificationShape::token_count(
  self : FusedQualificationShape,
) -> Int {
  self.token_count
}

///|
pub fn FusedQualificationShape::positions(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  self.positions
}

///|
pub fn FusedQualificationShape::page_table(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  self.page_table
}

///|
pub struct FusedQkvQualificationFixture {
  priv shape : FusedQualificationShape
  priv input : ReadOnlyArray[Int]
  priv q_weight : ReadOnlyArray[Int]
  priv k_weight : ReadOnlyArray[Int]
  priv v_weight : ReadOnlyArray[Int]
  priv rotated_qkv : ReadOnlyArray[Int]
  priv key_cache : ReadOnlyArray[Int]
  priv value_cache : ReadOnlyArray[Int]
}

///|
pub struct FusedResidualNormQualificationFixture {
  priv shape : FusedQualificationShape
  priv residual : ReadOnlyArray[Int]
  priv branch : ReadOnlyArray[Int]
  priv norm_weight : ReadOnlyArray[Int]
  priv residual_output : ReadOnlyArray[Int]
  priv norm_output : ReadOnlyArray[Int]
}
