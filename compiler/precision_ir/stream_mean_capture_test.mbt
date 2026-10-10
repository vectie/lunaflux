///|
test "stream means have distinct segmented output and stream input shapes" {
  let p = @precision_ir.StreamMeanCapturePrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=3,
    segment_count=3,
    maximum_output_bytes=73728L,
  )
  inspect(p.geometry(), content="(4, 4096, 3, 3)")
  inspect(p.input_bytes(), content="98304")
  inspect(p.output_bytes(), content="73728")
  for
    (streams, hidden, rows, segments) in [
      (0, 1, 1, 1),
      (1, 0, 1, 1),
      (1, 1, 0, 1),
      (1, 1, 1, 0),
    ] {
    try
      @precision_ir.StreamMeanCapturePrecision::new(
        stream_count=streams,
        hidden_width=hidden,
        maximum_rows=rows,
        segment_count=segments,
        maximum_output_bytes=100L,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      _ => fail("unexpected geometry error")
    } noraise {
      _ => fail("empty capture must fail")
    }
  }
  try
    @precision_ir.StreamMeanCapturePrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=3,
      segment_count=3,
      maximum_output_bytes=73727L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    _ => fail("unexpected budget error")
  } noraise {
    _ => fail("capture must obey the exact output budget")
  }
}
