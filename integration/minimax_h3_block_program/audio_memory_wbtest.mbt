///|
test "aggregate audio memory sum cannot reuse or overflow a ceiling" {
  assert_eq(audio_memory_sum([10L, 20L, 30L], 60L), 60L)
  for
    (parts, maximum) in [
      ([10L, 20L, 30L], 59L),
      ([9223372036854775807L, 1L], 9223372036854775807L),
      ([0L], 1L),
    ] {
    try audio_memory_sum(parts[:], maximum) catch {
      _ => ()
    } noraise {
      _ => fail("aggregate overbudget admitted")
    }
  }
}
