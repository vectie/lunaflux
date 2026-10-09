///|
test "seven stage chain reuses two buffers and rejects broken edges" {
  let shapes = [
    (16L, 32L),
    (32L, 80L),
    (80L, 80L),
    (80L, 80L),
    (80L, 80L),
    (80L, 80L),
    (80L, 80L),
  ]
  let layout = audio_chain_layout(shapes, 512L)
  assert_eq(layout.stride, 256L)
  assert_eq(layout.bytes, 512L)
  try audio_chain_layout(shapes, 511L) catch {
    _ => ()
  } noraise {
    _ => fail("short chain budget accepted")
  }
  try audio_chain_layout(shapes[:6], 512L) catch {
    _ => ()
  } noraise {
    _ => fail("incomplete chain accepted")
  }
  shapes[3] = (79L, 80L)
  try audio_chain_layout(shapes, 512L) catch {
    _ => ()
  } noraise {
    _ => fail("mismatched stage edge accepted")
  }
}
