///|
test "audio source contains real causal mean encoder and complete symbol table" {
  let source = @minimax_h3_media_encode.AudioSource::new(
    samples=801,
    max_scratch_bytes=2000000000L,
  )
  assert_eq(source.kernels.symbols.length(), source.plan.graph.nodes.length())
  assert_eq(source.kernels.grids.length(), source.plan.graph.nodes.length())
  let text = @utf8.decode(source.kernels.source)
  assert_true(text.contains("k<=q"))
  assert_true(text.contains("1e-9f"))
  assert_true(text.contains("1e-5f"))
  assert_true(text.contains("tanhf(inner)"))
  assert_false(text.contains("__nv_bfloat16"))
  assert_false(text.contains("curand"))
  for name in source.kernels.symbols {
    assert_true(text.contains("void " + name + "("))
  }
}

///|
test "empty request cannot submit poll or expose rows and closes idempotently" {
  let request = @minimax_h3_media_encode.AudioRequest::new()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("submitted empty request")
  }
  try request.poll() catch {
    _ => ()
  } noraise {
    _ => fail("polled empty request")
  }
  try request.with_rows((_, _, _) => ()) catch {
    _ => ()
  } noraise {
    _ => fail("exposed empty output")
  }
  request.cancel()
  request.close()
  request.close()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("submitted closed request")
  }
}
