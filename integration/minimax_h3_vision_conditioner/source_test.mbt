///|
test "vision source contains complete noncausal blocks and distinct merger activations" {
  let p = @vision.VisionPlan::new(
    [{ time: 2, height: 2, width: 4, }],
    max_workspace_bytes=10000000L,
    max_output_bytes=10000000L,
  )
  let source = build_source(p, prefix="vision")
  assert_eq(source.symbols.length(), 13)
  let s = @utf8.decode(source.source)
  for symbol in source.symbols {
    assert_true(s.contains("void " + symbol + "("))
  }
  assert_true(s.contains("for(int key=start;key<end;++key)"))
  assert_true(s.contains("tanhf("))
  assert_true(s.contains("erff("))
  assert_true(s.contains("row<16"))
  assert_true(s.contains("*start=0+(r/8)*8"))
  try build_source(p, prefix="bad;injection") catch {
    _ => ()
  } noraise {
    _ => fail("unsafe symbol accepted")
  }
}
