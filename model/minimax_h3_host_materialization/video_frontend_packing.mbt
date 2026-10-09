///|
pub fn MiniMaxH3HostComponentArena::resolve_video_frontend_weights(
  self : Self,
  packing : @minimax_h3_weights.VideoFrontendWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is VideoVae else { raise InvalidComponent }
  resolve_video_frontend_regions(self.tensors, packing)
}

///|
fn resolve_video_frontend_regions(
  tensors : ArrayView[MiniMaxH3HostTensor],
  packing : @minimax_h3_weights.VideoFrontendWeightPacking,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  ReadOnlyArray::from_array(
    packing.copies[:].map(copy => {
      let resolved = resolve_block_slice(
        tensors,
        copy.tensor_name,
        0L,
        copy.byte_count,
        copy.destination_offset,
      )
      for tensor in tensors {
        if tensor.name == copy.tensor_name &&
          tensor.byte_length != copy.byte_count {
          raise InvalidSlice(copy.tensor_name)
        }
      }
      resolved
    }),
  )
}
