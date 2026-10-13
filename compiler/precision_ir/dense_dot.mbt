///|
/// Matrix accumulation permits BF16 products accumulated in F32 by a matrix
/// primitive. It preserves input/output rounds, but is not the ordered scalar
/// reduction: reassociation must be selected explicitly and validated outside
/// the compiler. No reference implementation or device name is part of this law.
pub(all) enum DenseDotLaw {
  ScalarFloat32
  MatrixFloat32
} derive(Debug, Eq)

///|
pub extend DenseDotLaw with Eq::{equal, not_equal}

///|
pub extend DenseDotLaw with @debug.Debug::{to_repr}
