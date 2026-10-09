///|
test "vision weight resolver rejects equal byte count wrong shapes" {
  let packing = @vision_plan.frontend_weights()
  let tensors : Array[MiniMaxH3HostTensor] = packing.regions[:].map(r => {
    name: r.name,
    role: VisionPatchEmbedding,
    shape: r.shape,
    arena_index: 0,
    arena_offset: r.offset + 64L,
    byte_length: r.bytes,
  })
  let resolved = resolve_vision_regions(tensors, packing.regions[:])
  assert_eq(resolved.length(), 3)
  assert_eq(resolved[0].source_offset, 64L)
  assert_eq(resolved[0].destination_offset, 0L)
  let first = tensors[0]
  tensors[0] = { ..first, shape: [first.byte_length / 2L], }
  try resolve_vision_regions(tensors, packing.regions[:]) catch {
    InvalidSlice(_) => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("wrong shape accepted")
  }
}
