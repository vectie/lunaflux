///|
test "audio cache budget is aggregate and cannot wrap" {
  audio_cache_budget([10L, 20L, 30L], 60L)
  try audio_cache_budget([10L, 20L, 30L], 59L) catch {
    _ => ()
  } noraise {
    _ => fail("per-stage ceiling mistaken for total")
  }
  try
    audio_cache_budget([9223372036854775807L, 1L], 9223372036854775807L)
  catch {
    _ => ()
  } noraise {
    _ => fail("cache sum wrapped")
  }
  try audio_cache_budget([0L], 1L) catch {
    _ => ()
  } noraise {
    _ => fail("empty cache accepted")
  }
}
