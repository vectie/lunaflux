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
  resolve_video_source_regions(tensors, packing.copies[:])
}

///|
fn resolve_video_source_regions(
  tensors : ArrayView[MiniMaxH3HostTensor],
  copies : ArrayView[@minimax_h3_weights.BlockWeightCopy],
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  let original = tensors.any(t => t.name == "decoder.mask_token")
  let sources : Array[@minimax_h3_weights.BlockWeightCopy] = []
  for copy in copies {
    if original {
      sources.append(
        @minimax_h3_weights.original_video_vae_copies(copy) catch {
          _ => raise InvalidSlice(copy.tensor_name)
        },
      )
    } else {
      sources.push(copy)
    }
  }
  ReadOnlyArray::from_array(
    sources.map(copy => {
      let resolved = resolve_block_slice(
        tensors,
        copy.tensor_name,
        copy.source_offset,
        copy.byte_count,
        copy.destination_offset,
      )
      for tensor in tensors {
        if !original &&
          tensor.name == copy.tensor_name &&
          tensor.byte_length != copy.byte_count {
          raise InvalidSlice(copy.tensor_name)
        }
      }
      resolved
    }),
  )
}
