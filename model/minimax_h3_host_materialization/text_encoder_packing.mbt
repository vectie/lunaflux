///|
fn resolve_text_encoder_regions(
  tensors : ArrayView[MiniMaxH3HostTensor],
  regions : ArrayView[@text_plan.WeightRegion],
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  ReadOnlyArray::from_array(
    regions.map(r => {
      let copy = resolve_block_slice(tensors, r.name, 0L, r.bytes, r.offset)
      for tensor in tensors {
        if tensor.name == r.name && tensor.byte_length != r.bytes {
          raise InvalidSlice(r.name)
        }
      }
      copy
    }),
  )
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_text_encoder_layer(
  self : Self,
  packing : @text_plan.LayerWeights,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is TextConditioner else { raise InvalidComponent }
  resolve_text_encoder_regions(self.tensors, packing.regions[:])
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_text_encoder_embedding(
  self : Self,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is TextConditioner else { raise InvalidComponent }
  let region : @text_plan.WeightRegion = @text_plan.embedding_region()
  resolve_text_encoder_regions(self.tensors, [region])
}
