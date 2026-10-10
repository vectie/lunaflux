///|
test "video budget subtracts borrowed input statistics weights and module reserve" {
  assert_eq(video_workspace_limit(1000L, 500L, 100L, 192L, 100L), 108L)
  for
    (limit, weights, input, stats, reserve) in [
      (892L, 500L, 100L, 192L, 100L),
      (1000L, -1L, 100L, 192L, 100L),
      (9223372036854775807L, 9223372036854775707L, 101L, 192L, 100L),
    ] {
    try video_workspace_limit(limit, weights, input, stats, reserve) catch {
      _ => ()
    } noraise {
      _ => fail("invalid video budget accepted")
    }
  }
  assert_eq(
    video_statistics([0.0F, 1.0F, -1.0F]),
    b"\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x80\xbf",
  )
}
