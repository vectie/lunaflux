///|
/// Per-key online softmax differs from the tiled BF16 probability fold:
/// its probability and denominator remain F32, and statistics are published
/// to the owners of the output components before their ordered value update.
pub(all) enum KeyScaleStep {
  MergeKeyStatistics
  PublishKeyScales
} derive(Debug, Eq)

///|
pub extend KeyScaleStep with Eq::{equal, not_equal}

///|
pub extend KeyScaleStep with @debug.Debug::{to_repr}

///|
pub struct KeyFold {
  priv dimensions : Int
} derive(Debug, Eq)

///|
pub extend KeyFold with Eq::{equal, not_equal}

///|
pub extend KeyFold with @debug.Debug::{to_repr}

///|
pub fn KeyFold::new(dimensions~ : Int) -> KeyFold? {
  if dimensions > 0 {
    Some({ dimensions, })
  } else {
    None
  }
}

///|
pub fn KeyFold::scale_steps(_self : KeyFold) -> ReadOnlyArray[KeyScaleStep] {
  [MergeKeyStatistics, PublishKeyScales]
}

///|
/// Distribute output components without baking a device subgroup width into
/// the physical state law. The backend supplies its actual owner width.
pub fn KeyFold::values_per_owner(self : KeyFold, owners : Int) -> Int? {
  if owners <= 0 {
    None
  } else {
    Some(
      ((self.dimensions.to_int64() + owners.to_int64() - 1L) / owners.to_int64()).to_int(),
    )
  }
}
