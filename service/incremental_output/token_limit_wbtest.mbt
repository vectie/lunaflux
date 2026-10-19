///|
test "token budget cuts incomplete UTF-8 without weakening complete-text decoding" {
  for prefix in [[0xe4], [0xe4, 0xbd], [0xf0, 0x9f, 0x90]] {
    let workspace = wb_output_workspace(1)
    let work = workspace.begin(stops([]), special_policy=Preserve)
    wb_drive_output_work(work)
    let lease = work.take_lease()
    let destination = FixedArray::make(32, b'!')
    for token in prefix {
      let pushed = lease.push_token_status(token, destination, destination_offset=0)
      assert_true(pushed.is_ready())
      assert_eq(pushed.copied_bytes(), 0)
    }
    assert_false(lease.finish_into_status(destination, destination_offset=0).is_ready())
    let terminal = lease.finish_into_status(destination, destination_offset=0, boundary=TokenLimit)
    assert_true(terminal.is_ready())
    assert_eq(terminal.copied_bytes(), 0)
    assert_true(lease.is_finished())
    assert_eq(destination[0], b'!')
    assert_false(lease.finish_into_status(destination, destination_offset=0, boundary=TokenLimit).is_ready())
    lease.release()
  }
}

///|
test "token budget retains valid stop-prefix bytes and destination retry semantics" {
  let workspace = wb_output_workspace(1)
  let work = workspace.begin(stops([b"A\xe4\xbd\xa0"]), special_policy=Preserve)
  wb_drive_output_work(work)
  let lease = work.take_lease()
  let destination = FixedArray::make(32, b'!')
  for token in [65, 0xe4, 0xbd] {
    let pushed = lease.push_token_status(token, destination, destination_offset=0)
    assert_true(pushed.is_ready())
    assert_eq(pushed.copied_bytes(), 0)
  }
  assert_false(lease.finish_into_status(FixedArray::make(0, b'!'), destination_offset=0, boundary=TokenLimit).is_ready())
  let terminal = lease.finish_into_status(destination, destination_offset=1, boundary=TokenLimit)
  assert_true(terminal.is_ready())
  assert_eq(terminal.copied_bytes(), 1)
  assert_eq(destination[0], b'!')
  assert_eq(destination[1], b'A')
  assert_eq(destination[2], b'!')
  lease.release()
}

///|
test "token-limit finalization does not accept malformed token bytes" {
  let workspace = wb_output_workspace(1)
  let work = workspace.begin(stops([]), special_policy=Preserve)
  wb_drive_output_work(work)
  let lease = work.take_lease()
  let destination = FixedArray::make(32, b'!')
  assert_false(lease.push_token_status(0xc0, destination, destination_offset=0).is_ready())
  assert_true(lease.push_token_status(65, destination, destination_offset=0).is_ready())
  assert_eq(destination[0], b'A')
  assert_true(lease.finish_into_status(destination, destination_offset=1, boundary=TokenLimit).is_ready())
  lease.release()
}
