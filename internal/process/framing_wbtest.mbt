///|
test "oversized declared frame poisons alignment before payload read" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let declared = FixedArray::make(4, b'\x00')
  declared[1] = b'\x20'
  child.write_exact(declared, offset=0, byte_count=4)
  try child.read_frame(FixedArray::make(4096, b'\x00')) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected oversized frame error")
  } noraise {
    _ => fail("oversized declared frame was admitted")
  }
  assert_true(child.is_failed())
  child.close()
}
