///|
test "video frontend resolves all tensors and rejects partial slices" {
  let packing = @minimax_h3_weights.video_frontend_weight_packing(
    max_bytes=239968L,
  )
  let tensors : Array[MiniMaxH3HostTensor] = packing.copies[:].map(copy => {
    name: copy.tensor_name,
    role: VideoDecoder,
    shape: ReadOnlyArray::from_array([copy.byte_count / 4L]),
    arena_index: 2,
    arena_offset: copy.destination_offset + 128L,
    byte_length: copy.byte_count,
  })
  let resolved = resolve_video_frontend_regions(tensors, packing)
  assert_eq(resolved.length(), 5)
  for i, copy in resolved {
    assert_eq(copy.arena_index, 2)
    assert_eq(copy.source_offset, packing.copies[i].destination_offset + 128L)
    assert_eq(copy.destination_offset, packing.copies[i].destination_offset)
    assert_eq(copy.byte_count, packing.copies[i].byte_count)
  }
  try resolve_video_frontend_regions(tensors[1:], packing) catch {
    MissingTensor(_) => ()
    _ => fail("incorrect missing tensor error")
  } noraise {
    _ => fail("missing post-quant tensor accepted")
  }
  let first = tensors[0]
  tensors[0] = { ..first, byte_length: first.byte_length + 4L, }
  try resolve_video_frontend_regions(tensors, packing) catch {
    InvalidSlice(_) => ()
    _ => fail("incorrect slice error")
  } noraise {
    _ => fail("partial source tensor accepted")
  }
}

///|
test "host and streamed decoder packing share original head-interleaved slices" {
  let manifest = video_vae_manifest_fixture(original=true)
  let tensors = Array::makei(manifest.tensor_count(), i => manifest.tensor(i))
  let packing = @minimax_h3_weights.video_block_weight_packing(
    layer=0,
    max_bytes=300000000L,
  )
  let actual = resolve_video_source_regions(tensors, packing.copies[:])
  let expected : Array[@minimax_h3_weights.BlockWeightCopy] = []
  for copy in packing.copies {
    expected.append(@minimax_h3_weights.original_video_vae_copies(copy))
  }
  assert_eq(actual.length(), expected.length())
  assert_eq(actual.length(), packing.copies.length() + 6 * 31)
  for i, copy in expected {
    let source = tensors.filter(t => t.name() == copy.tensor_name)[0]
    assert_eq(
      actual[i].source_offset,
      source.arena_offset() + copy.source_offset,
    )
    assert_eq(actual[i].byte_count, copy.byte_count)
    assert_eq(actual[i].destination_offset, copy.destination_offset)
  }
  let frontend = @minimax_h3_weights.video_frontend_weight_packing(
    max_bytes=1000000L,
  )
  assert_eq(resolve_video_frontend_regions(tensors, frontend).length(), 5)
}
