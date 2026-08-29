///|
#borrow(child, path)
extern "c" fn raw_test_spawn_fixture(child : NativeChild, path : Bytes) -> Int = "lunaflux_process_test_spawn_fixture"

///|
fn fixture_child_owner(limits : ProcessLimits) -> ChildProcess {
  ignore(@native_fixture.linked())
  let empty = FixedArray::make(0, b'\x00')
  {
    handle: raw_prepare_child(),
    limits,
    write_prefix: FixedArray::make(4, b'\x00'),
    read_prefix: FixedArray::make(4, b'\x00'),
    empty_transfer_buffer: empty,
    write_transfer_active: false,
    write_transfer_source: empty,
    write_transfer_offset: 0,
    write_transfer_byte_count: 0,
    write_transfer_deadline_millis: 0L,
    write_transfer_phase: 0,
    write_transfer_cursor: 0,
    read_transfer_active: false,
    read_transfer_destination: empty,
    read_transfer_staging: empty,
    read_transfer_byte_count: 0,
    read_transfer_deadline_millis: 0L,
    read_transfer_phase: 0,
    read_transfer_cursor: 0,
    maintenance_phase: 0,
    maintenance_phase_started_millis: 0L,
    maintenance_deadline_millis: 0L,
    maintenance_resume_phase: 0,
    maintenance_exit_kind: 0,
    maintenance_exit_code: 0,
    maintenance_has_exit: false,
    failed: false,
    closed: true,
  }
}

///|
fn spawn(
  path : Bytes,
  limits : ProcessLimits,
) -> ChildProcess raise ProcessError {
  let child = fixture_child_owner(limits)
  if raw_test_spawn_fixture(child.handle, path) != 0 {
    raise SpawnFailed
  }
  child.closed = false
  child
}
