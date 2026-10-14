///|
/// Matrix accumulation permits BF16 products accumulated in F32 by a matrix
/// primitive. It preserves input/output rounds, but is not the ordered scalar
/// reduction: reassociation must be selected explicitly and validated outside
/// the compiler. No reference implementation or device name is part of this law.
pub(all) enum DenseDotLaw {
  ScalarFloat32
  MatrixFloat32
  // Explicitly permits a deterministic partitioned F32 reduction for vector
  // shapes and matrix accumulation for multiple rows. Publication is unchanged.
  // This is a different numerical program, not a rewrite of ScalarFloat32.
  ShapeAwareFloat32
} derive(Debug, Eq)

///|
pub extend DenseDotLaw with Eq::{equal, not_equal}

///|
pub extend DenseDotLaw with @debug.Debug::{to_repr}
