///|
pub fn MiniMaxH3HostComponentArena::resolve_context_weights(
  self : Self,
  packing : @minimax_h3_weights.ContextWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}
