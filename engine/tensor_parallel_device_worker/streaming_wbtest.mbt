///|
test "device-resident prepared rank is idle and never grants streaming ownership" {
  let residency = RankKvResidency::DeviceResident
  assert_true(residency.is_idle())
  residency.close()
  residency.close()
  try residency.streaming() catch {
    Invalid(KvBinding) => ()
    _ => fail("unexpected residency failure")
  } noraise {
    _ => fail("device-resident rank granted a streaming owner")
  }
}
