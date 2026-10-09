///|
pub struct ResolvedBlockWeightCopy {
  arena_index : Int
  source_offset : Int64
  destination_offset : Int64
  byte_count : Int64
}

///|
pub(all) suberror BlockWeightResolutionError {
  Unavailable
  IdentityMismatch
  InvalidComponent
  MissingTensor(String)
  InvalidSlice(String)
}

///|
fn resolve_block_slice(
  tensors : ArrayView[MiniMaxH3HostTensor],
  name : String,
  offset : Int64,
  bytes : Int64,
  destination : Int64,
) -> ResolvedBlockWeightCopy raise BlockWeightResolutionError {
  for tensor in tensors {
    if tensor.name == name {
      guard offset >= 0L &&
        bytes > 0L &&
        offset <= tensor.byte_length &&
        bytes <= tensor.byte_length - offset else {
        raise InvalidSlice(name)
      }
      return {
        arena_index: tensor.arena_index,
        source_offset: tensor.arena_offset + offset,
        destination_offset: destination,
        byte_count: bytes,
      }
    }
  }
  raise MissingTensor(name)
}

///|
/// Resolve once before device upload. No payload copying and no GPU work.
/// The returned offsets remain valid only while this host arena is retained.
pub fn MiniMaxH3HostComponentArena::resolve_block_weights(
  self : Self,
  packing : @minimax_h3_weights.BlockWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_output_weights(
  self : Self,
  packing : @minimax_h3_weights.OutputWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_timestep_weights(
  self : Self,
  packing : @minimax_h3_weights.TimestepWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_input_weights(
  self : Self,
  packing : @minimax_h3_weights.InputWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}

///|
pub fn MiniMaxH3HostComponentArena::resolve_refiner_weights(
  self : Self,
  packing : @minimax_h3_weights.RefinerWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  self.resolve_weight_copies(packing.copies[:], expected_identity~)
}

///|
fn MiniMaxH3HostComponentArena::resolve_weight_copies(
  self : Self,
  copies : ArrayView[@minimax_h3_weights.BlockWeightCopy],
  expected_identity~ : @spec.ModelIdentity,
) -> ReadOnlyArray[ResolvedBlockWeightCopy] raise BlockWeightResolutionError {
  guard !self.released else { raise Unavailable }
  guard self.identity == expected_identity else { raise IdentityMismatch }
  guard self.component is (Fl2vaDenoiser | Ref2vaDenoiser) else {
    raise InvalidComponent
  }
  let resolved = copies.map(copy => {
    resolve_block_slice(
      self.tensors,
      copy.tensor_name,
      copy.source_offset,
      copy.byte_count,
      copy.destination_offset,
    )
  })
  ReadOnlyArray::from_array(resolved)
}
