///|
pub fn MiniMaxH3HostComponentArena::resolve_audio_decoder_weights(
  self : Self,
  packing : @minimax_h3_weights.AudioDecoderWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is AudioVae else { raise InvalidComponent }
  ReadOnlyArray::from_array(
    packing.copies[:].map(copy => {
      resolve_audio_whole_tensor(
        self.tensors,
        copy.tensor_name,
        copy.byte_count,
        copy.destination_offset,
      )
    }),
  )
}

///|
fn resolve_audio_whole_tensor(
  tensors : ArrayView[MiniMaxH3HostTensor],
  name : String,
  bytes : Int64,
  destination : Int64,
) -> ResolvedBlockWeightCopy raise BlockWeightResolutionError {
  let resolved = resolve_block_slice(tensors, name, 0L, bytes, destination)
  for tensor in tensors {
    if tensor.name == name && tensor.byte_length != bytes {
      raise InvalidSlice(name)
    }
  }
  resolved
}
