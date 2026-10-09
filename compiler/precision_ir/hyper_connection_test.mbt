///|
test "hyper connection precision accounts six frames without device schedules" {
  let plan = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=32,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.00001,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  assert_eq(plan.geometry(), (4, 4096, 32))
  assert_eq(plan.frame_bytes(), [512L, 2048L, 2048L, 262144L, 262144L, 262144L])
  assert_eq(plan.workspace_bytes(), 791040L)
  assert_eq(plan.sinkhorn_iterations(), 20)
  assert_true(plan.mix_law() is RoundedBf16ControlsAndProducts)
}

///|
test "hyper connection precision rejects overflow and invalid arithmetic" {
  for
    (streams, hidden, rows, epsilon) in [
      (1, 32, 1, 0.00001),
      (4, 32, 0, 0.00001),
      (4, 32, 1, 0.0),
      (2147483647, 2147483647, 2147483647, 0.00001),
    ] {
    try
      @precision_ir.HyperConnectionPrecision::new(
        stream_count=streams,
        hidden_width=hidden,
        maximum_rows=rows,
        sinkhorn_iterations=20,
        control_epsilon=0.000001,
        normalization_epsilon=epsilon,
        mix_law=OrderedF32SingleBf16Round,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid hyper geometry accepted")
    }
  }
}
