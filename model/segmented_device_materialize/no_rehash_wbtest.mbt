///|
// A package-private synthetic label proves that preparation uses the retained
// label instead of regenerating it. Public callers cannot forge these fields.
test "all upload modes consume immutable labels without layout rehash" {
  let original = segmented_test_layout()
  let layout = { ..original, digest: { hex: "e".repeat(64), }, }
  let arenas = [FixedArray::make(8, b'a'), FixedArray::make(8, b'b')]
  validate_layout_replay(
    layout,
    arenas,
    layout.source_identity(),
    layout.source_digest(),
    layout.digest(),
  )
  validate_borrowed_source(layout, provide=(ordinal, consumer) => {
    consumer.accept(arenas[ordinal])
  })
  let streamed = prepare_with_streamed_resource(
    layout,
    bytes => {
      fake_segmented_allocation(
        bytes,
        fail_validate_at=-1,
        fail_copy_at=-1,
        close_failures=0,
      )
    },
    fake_validate_segmented,
    _ => (),
    fake_close_segmented,
  )
  guard streamed is SegmentedResourceReady(ready) else {
    fail("streamed label rejected")
  }
  assert_eq(ready.resource.validate_calls, 3)
  fake_close_segmented(ready.resource)
  let borrowed = prepare_with_borrowed_resource(
    layout,
    layout.source_identity(),
    layout.source_digest(),
    layout.digest(),
    (ordinal, consumer) => consumer.accept(arenas[ordinal]),
    bytes => {
      fake_segmented_allocation(
        bytes,
        fail_validate_at=-1,
        fail_copy_at=-1,
        close_failures=0,
      )
    },
    fake_validate_segmented,
    fake_copy_segmented,
    fake_close_segmented,
  )
  guard borrowed is SegmentedResourceReady(copied) else {
    fail("borrowed label rejected")
  }
  assert_eq(copied.resource.copy_calls, 3)
  assert_eq(Bytes::from_array(copied.resource.bytes[0:3]), b"aaa")
  fake_close_segmented(copied.resource)
  try
    validate_layout_replay(
      layout,
      arenas,
      layout.source_identity(),
      layout.source_digest(),
      original.digest(),
    )
  catch {
    LayoutDigestMismatch => ()
    _ => fail("incorrect plan-association failure")
  } noraise {
    _ => fail("unrelated plan label accepted")
  }
}
