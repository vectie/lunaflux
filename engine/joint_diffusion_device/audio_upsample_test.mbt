///|
test "seven audio upsamplers preserve exact time and weight norm axes" {
  for i, stride in [5, 5, 2, 2, 2, 2, 2]; time = 207 {
    let ci = 1024 >> i
    let co = ci / 2
    let kernel = if stride == 5 { 9 } else { 4 }
    let layout = @joint_diffusion_device.AudioUpsampleLayout::new(
      batch=2,
      input_channels=ci,
      output_channels=co,
      input_time=time,
      kernel~,
      stride~,
      padding=(kernel - stride) / 2,
      max_weight_bytes=100000000L,
    )
    assert_eq(layout.output_time, time * stride)
    assert_eq(layout.g_bytes, ci.to_int64() * 4L)
    assert_eq(layout.bias_bytes, co.to_int64() * 4L)
    assert_eq(
      layout.weight_bytes,
      ci.to_int64() * co.to_int64() * kernel.to_int64() * 4L,
    )
    continue layout.output_time
  } nobreak {
    assert_eq(time, 165600)
  }
  try
    @joint_diffusion_device.AudioUpsampleLayout::new(
      batch=2,
      input_channels=8,
      output_channels=4,
      input_time=1,
      kernel=4,
      stride=2,
      padding=2,
      max_weight_bytes=10000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("empty output accepted")
  }
}
