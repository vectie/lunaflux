///|
extern "c" fn raw_test_expect_clean_eof(mode : Int) -> Int = "lunaflux_process_test_expect_clean_eof"

///|
extern "c" fn raw_test_read_prefix_or_eof(mode : Int) -> Int = "lunaflux_process_test_read_prefix_or_eof"

///|
extern "c" fn raw_test_begin_inherited_frame(mode : Int) -> Int = "lunaflux_process_test_begin_inherited_frame"

///|
extern "c" fn raw_test_restore_stdin(saved_stdin : Int) -> Int = "lunaflux_process_test_restore_stdin"

///|
test "inherited clean EOF rejects bytes partial prefixes and timeout" {
  assert_eq(raw_test_expect_clean_eof(0), 0)
  assert_eq(raw_test_expect_clean_eof(1), 4)
  assert_eq(raw_test_expect_clean_eof(2), 4)
  assert_eq(raw_test_expect_clean_eof(3), 2)
}

///|
fn[T] with_inherited_frame(mode : Int, body : () -> T raise) -> T raise {
  let saved = raw_test_begin_inherited_frame(mode)
  if saved < 0 {
    fail("failed to install inherited frame fixture")
  }
  try {
    let result = body()
    assert_eq(raw_test_restore_stdin(saved), 0)
    result
  } catch {
    error => {
      assert_eq(raw_test_restore_stdin(saved), 0)
      raise error
    }
  }
}

///|
test "public inherited frame-or-EOF accepts clean close and complete frame" {
  let limits = ProcessLimits::new(
    max_frame_bytes=64,
    io_timeout_millis=20,
    shutdown_timeout_millis=20,
  )
  assert_eq(
    with_inherited_frame(0, () => {
      open_inherited(limits).read_frame_or_eof(FixedArray::make(64, b'\x00'))
    }),
    0,
  )
  with_inherited_frame(1, () => {
    let input = FixedArray::make(64, b'\x00')
    assert_eq(open_inherited(limits).read_frame_or_eof(input), 2)
    assert_eq(input[0], b'\x07')
    assert_eq(input[1], b'\x09')
  })
}

///|
test "public inherited frame-or-EOF poisons partial prefix and payload" {
  let limits = ProcessLimits::new(
    max_frame_bytes=64,
    io_timeout_millis=20,
    shutdown_timeout_millis=20,
  )
  for mode in [2, 3] {
    with_inherited_frame(mode, () => {
      let channel = open_inherited(limits)
      let input = FixedArray::make(64, b'\x00')
      try channel.read_frame_or_eof(input) catch {
        OperationFailed(Read) | ChannelClosed(Read) => ()
        _ => fail("unexpected partial inherited frame error")
      } noraise {
        _ => fail("partial inherited frame was accepted")
      }
      assert_true(channel.is_failed())
      try channel.read_frame_or_eof(input) catch {
        InvalidState(Read) => ()
        _ => fail("poisoned inherited channel returned unexpected error")
      } noraise {
        _ => fail("poisoned inherited channel remained usable")
      }
    })
  }
}

///|
test "inherited frame prefix distinguishes clean EOF from partial EOF" {
  assert_eq(raw_test_read_prefix_or_eof(0), 3)
  assert_eq(raw_test_read_prefix_or_eof(1), 0)
  assert_eq(raw_test_read_prefix_or_eof(2), 4)
  assert_eq(raw_test_read_prefix_or_eof(3), 2)
}
