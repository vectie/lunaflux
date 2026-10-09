///|
test "AMP resolution covers all tensors and rejects partial tensor matches" {
  let layout = @minimax_h3_vae_reference.bind_audio_decoder_amp_parameter_layout(
    stage_index=6,
  )
  let tensors : Array[MiniMaxH3HostTensor] = layout.regions()[:].map(region => {
    name: region.tensor_name(),
    role: DenoiserFeedForward,
    shape: ReadOnlyArray::from_array([region.element_count()]),
    arena_index: 3,
    arena_offset: region.byte_offset() + 256L,
    byte_length: region.byte_count(),
  })
  let copies = resolve_audio_amp_regions(tensors, layout)
  assert_eq(copies.length(), 126)
  for i, copy in copies {
    assert_eq(copy.arena_index, 3)
    assert_eq(copy.source_offset, layout.regions()[i].byte_offset() + 256L)
    assert_eq(copy.destination_offset, layout.regions()[i].byte_offset())
    assert_eq(copy.byte_count, layout.regions()[i].byte_count())
  }
  try resolve_audio_amp_regions(tensors[1:], layout) catch {
    MissingTensor(_) => ()
    _ => fail("wrong missing AMP tensor error")
  } noraise {
    _ => fail("missing AMP parameter accepted")
  }
  let first = tensors[0]
  tensors[0] = { ..first, byte_length: first.byte_length + 4L, }
  try resolve_audio_amp_regions(tensors, layout) catch {
    InvalidSlice(_) => ()
    _ => fail("wrong oversized AMP tensor error")
  } noraise {
    _ => fail("partial AMP tensor accepted")
  }
}
