///|
test "VAE read write regions reject overlap without end arithmetic overflow" {
  assert_true(vae_regions_overlap(0L, 16L, 8L, 16L))
  assert_true(vae_regions_overlap(8L, 16L, 0L, 16L))
  assert_true(vae_regions_overlap(8L, 16L, 8L, 16L))
  assert_false(vae_regions_overlap(0L, 16L, 16L, 16L))
  assert_false(vae_regions_overlap(16L, 16L, 0L, 16L))
  assert_false(vae_regions_overlap(0L, 16L, 9223372036854775806L, 1L))
  assert_true(
    vae_regions_overlap(9223372036854775806L, 1L, 9223372036854775806L, 1L),
  )
}
