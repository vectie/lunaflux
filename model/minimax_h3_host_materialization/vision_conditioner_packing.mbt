///|
pub fn MiniMaxH3HostComponentArena::resolve_vision_conditioner_weights(
  self : Self,
  packing : @vision_plan.WeightGroup,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is TextConditioner else { raise InvalidComponent }
  resolve_vision_regions(self.tensors, packing.regions[:])
}

///|
fn resolve_vision_regions(
  tensors : ArrayView[MiniMaxH3HostTensor],
  regions : ArrayView[@vision_plan.WeightRegion],
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  ReadOnlyArray::from_array(
    regions.map(r => {
      let copy = resolve_block_slice(tensors, r.name, 0L, r.bytes, r.offset)
      for tensor in tensors {
        if tensor.name == r.name &&
          (tensor.byte_length != r.bytes || tensor.shape != r.shape) {
          raise InvalidSlice(r.name)
        }
      }
      copy
    }),
  )
}
