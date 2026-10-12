///|
/// Reassociation is an explicit numerical policy, separate from P storage.
/// TiledFloat32 permits matrix reduction and parallel softmax sums in F32;
/// it does not promise the bit pattern of an ascending scalar reduction.
/// Device instructions and physical tile dimensions belong to lowering.
pub(all) enum AttentionAccumulationLaw {
  OrderedFloat32
  TiledFloat32
} derive(Eq, Debug)

///|
pub extend AttentionAccumulationLaw with Eq::{equal, not_equal}

///|
pub extend AttentionAccumulationLaw with @debug.Debug::{to_repr}
