///|
test "rank-group process limits reject invalid buffer and timeout bounds" {
  try
    @rank_group_process.RankGroupProcessLimits::new(
      max_payload_bytes=0,
      startup_io_timeout_millis=1,
      io_timeout_millis=1,
      shutdown_timeout_millis=1,
      collective_exchange_timeout_millis=1,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected public limit rejection")
  } noraise {
    _ => fail("zero payload capacity was admitted")
  }
}
