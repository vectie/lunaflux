///|
test "unknown barrier count cannot satisfy a required barrier budget" {
  let observation = MachineWork::new(instructions=100L, local_bytes=0L).unwrap()
  let unconstrained = MachineWork::new(instructions=100L, local_bytes=0L).unwrap()
  let constrained = MachineWork::new(
    instructions=100L,
    local_bytes=0L,
    barriers=Some(0L),
  ).unwrap()
  assert_true(observation.fits(unconstrained))
  assert_false(observation.fits(constrained))
}
