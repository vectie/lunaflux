///|
test "logical component arena stays within small and large component ceilings" {
  for
    bytes in [
      1L, 605429308L, 1073741824L, 2147483584L, 2147483648L, 103079215104L,
    ] {
    // The host limits constructor rejects arena > total and arena > Int max.
    // Audio used to fail here before any device or request could be prepared.
    component_host_limits(1087, bytes) |> ignore
  }
  try component_host_limits(1087, 0L) catch {
    _ => ()
  } noraise {
    _ => fail("empty component ceiling admitted")
  }
}
