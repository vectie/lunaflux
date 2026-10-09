///|
test "compute progresses during NCCL submission but communication events wait" {
  let recorded = FixedArray::make(2, false)
  assert_true(parallel_command_ready(Kernel(1), recorded, false, true, true))
  assert_false(parallel_command_ready(Record(1), recorded, true, true, true))
  assert_false(parallel_command_ready(Wait(1), recorded, false, true, true))
  // NCCL has posted its GPU work. Record and dependent compute wait may now
  // be issued, even though the collective has not completed on the device.
  assert_true(parallel_command_ready(Record(1), recorded, true, true, false))
  recorded[1] = true
  assert_true(parallel_command_ready(Wait(1), recorded, false, true, false))
  // A next-collective dependency must not extend the stream being polled.
  assert_false(parallel_command_ready(Wait(1), recorded, true, true, false))
  assert_true(parallel_command_ready(Wait(1), recorded, true, false, false))
}
