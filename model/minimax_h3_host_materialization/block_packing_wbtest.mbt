///|
test "block packing resolves half tensors without payload allocation" {
  let tensors : Array[MiniMaxH3HostTensor] = [
    {
      name: "ff",
      role: DenoiserFeedForward,
      shape: ReadOnlyArray::from_array([8L, 4L]),
      arena_index: 2,
      arena_offset: 256L,
      byte_length: 64L,
    },
  ]
  let upper = resolve_block_slice(tensors, "ff", 32L, 32L, 0L)
  let lower = resolve_block_slice(tensors, "ff", 0L, 32L, 32L)
  assert_eq(upper.arena_index, 2)
  assert_eq(upper.source_offset, 288L)
  assert_eq(lower.source_offset, 256L)
  assert_eq(lower.destination_offset, 32L)
  for
    (offset, bytes) in [
      (-1L, 32L),
      (33L, 32L),
      (0L, 0L),
      (9223372036854775807L, 2L),
    ] {
    try resolve_block_slice(tensors, "ff", offset, bytes, 0L) catch {
      InvalidSlice(_) => ()
      _ => fail("wrong slice error")
    } noraise {
      _ => fail("invalid slice accepted")
    }
  }
  try resolve_block_slice(tensors, "absent", 0L, 1L, 0L) catch {
    MissingTensor(_) => ()
    _ => fail("wrong missing tensor error")
  } noraise {
    _ => fail("missing tensor accepted")
  }
}
