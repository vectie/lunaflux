///|
test "streamed uploader preserves segmented device layout without source arenas" {
  let layout = segmented_test_layout()
  let uploaded = Ref(0)
  let prepared = prepare_with_streamed_resource(
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
    allocation => {
      assert_eq(allocation.validate_calls, 3) catch {
        _ => raise HostSourceUnavailable(-1)
      }
      allocation.bytes[128] = b'x'
      uploaded.val += 1
    },
    fake_close_segmented,
  )
  guard prepared is SegmentedResourceReady(ready) else { fail("upload failed") }
  assert_eq(ready.layout, layout)
  assert_eq(ready.resource.bytes[128], b'x')
  assert_eq(uploaded.val, 1)
  fake_close_segmented(ready.resource)
}

///|
test "streamed failure retains cleanup owner only when close cannot complete" {
  let layout = segmented_test_layout()
  let result = prepare_with_streamed_resource(
    layout,
    bytes => {
      fake_segmented_allocation(
        bytes,
        fail_validate_at=-1,
        fail_copy_at=-1,
        close_failures=1,
      )
    },
    fake_validate_segmented,
    _ => raise HostSourceUnavailable(0),
    fake_close_segmented,
  )
  guard result is SegmentedResourceCleanupRequired(failed) else {
    fail("lost partially loaded owner")
  }
  assert_eq(failed.stage, Transfer)
  assert_eq(failed.cause, HostSourceUnavailable(0))
  assert_false(failed.resource.closed)
  fake_close_segmented(failed.resource)
  assert_true(failed.resource.closed)
}
