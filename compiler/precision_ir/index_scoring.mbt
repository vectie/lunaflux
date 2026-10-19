///|
/// Distinct finite-precision index programs. The matrix alternative quantizes
/// each complete BF16 query head and pooled key independently to finite E4M3
/// (RNE, saturate finite), using max(max_abs, 1e-12)/448 as its F32 scale.
/// Products accumulate in F32 before query/key rescaling, ReLU, dimension
/// scaling and the ordered head-weight fold. It is NOT an exact rewrite of
/// OrderedBFloat16 and requires external model-quality evaluation.
pub(all) enum IndexScoringLaw {
  OrderedBFloat16
  RowScaledE4M3Matrix
} derive(Eq)

///|
pub extend IndexScoringLaw with Eq::{equal, not_equal}

///|
pub fn IndexScoringLaw::pooled_row_bytes(self : Self, width : Int) -> Int64 {
  match self {
    OrderedBFloat16 => width.to_int64() * 2L
    RowScaledE4M3Matrix => width.to_int64() + 4L
  }
}
