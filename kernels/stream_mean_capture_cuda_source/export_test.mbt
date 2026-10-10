///|
test "export the actual generic capture source for independent GPU testing" {
  if @env.get_env_var("LUNA_EXPORT_STREAM_CAPTURE") != Some("1") {
    return
  }
  let plan = @precision.StreamMeanCapturePrecision::new(
    stream_count=4,
    hidden_width=257,
    maximum_rows=3,
    segment_count=3,
    maximum_output_bytes=4626L,
  )
  println("LUNA_CAPTURE_BEGIN")
  println(
    @utf8.decode(
      @stream_mean_capture_cuda_source.stream_mean_capture_source(
        plan,
        symbol="capture",
      ).source_bytes(),
    ),
  )
  println("LUNA_CAPTURE_END")
}
