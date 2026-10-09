///|
test "audio convolution extents cover all AMP widths and kernels" {
  for channels in [512, 256, 128, 64, 32, 16, 8] {
    for kernel in [3, 7, 11] {
      let bytes = channels.to_int64() *
        channels.to_int64() *
        kernel.to_int64() *
        4L
      let layout = @joint_diffusion_device.AudioConvolutionLayout::new(
        batch=2,
        channels~,
        time=1035,
        kernel~,
        max_weight_bytes=bytes,
      )
      assert_eq(layout.weight_bytes, bytes)
      assert_eq(layout.value_bytes, 2L * channels.to_int64() * 1035L * 4L)
      assert_eq(layout.channel_bytes, channels.to_int64() * 4L)
      try
        @joint_diffusion_device.AudioConvolutionLayout::new(
          batch=2,
          channels~,
          time=1035,
          kernel~,
          max_weight_bytes=bytes - 1L,
        )
      catch {
        _ => ()
      } noraise {
        _ => fail("short weight allocation accepted")
      }
    }
  }
  for
    (batch, channels, time, kernel) in [
      (0, 8, 10, 3),
      (2, 0, 10, 3),
      (2, 8, 0, 3),
      (2, 8, 10, 1),
      (2147483647, 2147483647, 2147483647, 11),
    ] {
    try
      @joint_diffusion_device.AudioConvolutionLayout::new(
        batch~,
        channels~,
        time~,
        kernel~,
        max_weight_bytes=9223372036854775807L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid convolution geometry accepted")
    }
  }
}
