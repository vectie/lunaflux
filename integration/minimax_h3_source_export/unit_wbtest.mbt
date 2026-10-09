///|
test "offline inventory refuses missing or repeated exported symbols" {
  for name in ["../escape", "a/b", "a\\b", "a\nmodule=bad", ".", ""] {
    try unit(name, b"extern void foo(){}", ["foo"], []) catch {
      _ => ()
    } noraise {
      _ => fail("unsafe module name accepted")
    }
  }
  try unit("a", b"extern void foo(){}", ["missing"], []) catch {
    InvalidModule => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("missing symbol accepted")
  }
  try unit("a", b"extern void foo(){}", ["foo", "foo"], []) catch {
    InvalidModule => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("duplicate symbol accepted")
  }
  let first = unit("a", b"extern void foo(){}", ["foo"], [("rows", 2L)])
  let second = unit("b", b"extern void foo(){}", ["foo"], [("rows", 2L)])
  assert_eq(first.sha256, second.sha256)
  // Fixed-symbol ABI reuse across two separately compiled modules is legal.
  let bundle : SourceExport = {
    model_digest: "test-only",
    units: [first],
    timestep_counts: [1],
  }
  assert_eq(bundle.append(second).units.length(), 2)
}
