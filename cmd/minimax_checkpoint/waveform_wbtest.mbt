///|
test "waveform diagnostic distinguishes finite extrema from nonfinite payload" {
  assert_eq(
    waveform_statistics(b"\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x00\xc0"),
    (0, 2.0F),
  )
  assert_eq(
    waveform_statistics(b"\x00\x00\x80\x7f\x00\x00\xc0\x7f\x00\x00\x80\xff"),
    (3, 0.0F),
  )
}
