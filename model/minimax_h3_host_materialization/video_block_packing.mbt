///|
pub fn MiniMaxH3HostComponentArena::resolve_video_block_weights(
  self : Self,
  packing : @minimax_h3_weights.VideoBlockWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is VideoVae else { raise InvalidComponent }
  resolve_video_source_regions(self.tensors, packing.copies[:])
}
