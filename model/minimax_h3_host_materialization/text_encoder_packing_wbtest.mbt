///|
test "text layer resolution preserves exact checkpoint up gate order and rejects truncation" {
  let layout = @text_plan.layer_weights(49)
  let tensors : Array[MiniMaxH3HostTensor] = layout.regions[:].map(r => {
    name: r.name,
    role: DenoiserFeedForward,
    shape: [r.bytes / 2L],
    arena_index: 2,
    arena_offset: r.offset + 128L,
    byte_length: r.bytes,
  })
  let copies = resolve_text_encoder_regions(tensors, layout.regions[:])
  assert_eq(copies.length(), 11)
  for i, copy in copies {
    assert_eq(copy.source_offset, layout.regions[i].offset + 128L)
    assert_eq(copy.destination_offset, layout.regions[i].offset)
    assert_eq(copy.byte_count, layout.regions[i].bytes)
  }
  try resolve_text_encoder_regions(tensors[1:], layout.regions[:]) catch {
    MissingTensor(_) => ()
    _ => fail("wrong missing error")
  } noraise {
    _ => fail("missing norm admitted")
  }
  let first = tensors[0]
  tensors[0] = { ..first, byte_length: first.byte_length + 2L, }
  try resolve_text_encoder_regions(tensors, layout.regions[:]) catch {
    InvalidSlice(_) => ()
    _ => fail("wrong extent error")
  } noraise {
    _ => fail("oversized tensor admitted")
  }
}
