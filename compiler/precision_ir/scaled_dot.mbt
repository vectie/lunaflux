///|
/// Keep scale application outside raw block accumulation when required by the
/// numeric contract. Matrix accumulation is an arithmetic choice, not a
/// promise of a particular device instruction or execution geometry.
pub(all) enum ScaledDotReductionLaw {
  DecodeOperandsFloat32
  UnscaledBlockMatrixFloat32(
    block_columns~ : Int,
    publication~ : Float32AffineLaw
  )
} derive(Debug, Eq)

///|
pub extend ScaledDotReductionLaw with Eq::{equal, not_equal}

///|
pub extend ScaledDotReductionLaw with @debug.Debug::{to_repr}
