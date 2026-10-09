///|
test "complete official audio encoder has deterministic mean path and exact checkpoint names" {
  let model = @minimax_h3_media_encode.AudioEncoder::new(
    samples=165600,
    max_scratch_bytes=2000000000L,
  )
  assert_eq(model.latent_time, 207)
  let graph = model.graph
  assert_eq(graph.values[graph.output].elements, 2L * 207L * 32L)
  let mut conv = 0
  let mut snake = 0
  let mut causal = 0
  let mut norm = 0
  for node in graph.nodes {
    match node.operation {
      Conv(..) => conv += 1
      Snake(..) => snake += 1
      CausalAttention(..) => causal += 1
      LayerNorm(..) => norm += 1
      _ => ()
    }
  }
  assert_eq(conv, 37)
  assert_eq(snake, 36)
  assert_eq(causal, 1)
  assert_eq(norm, 4)
  let mut mean = false
  for value in graph.values {
    if value.origin is Weight(name) {
      assert_false(name.has_prefix("logs_proj"))
      assert_false(name.has_prefix("decoder"))
      if name == "mean_proj.weight" {
        mean = true
      }
    }
  }
  assert_true(mean)
  assert_true(graph.scratch_bytes < 1000000000L)
}

///|
test "encoder rounds only to 800-sample hop and rejects empty input" {
  for (samples, time) in [(1, 1), (800, 1), (801, 2), (165601, 208)] {
    let model = @minimax_h3_media_encode.AudioEncoder::new(
      samples~,
      max_scratch_bytes=2000000000L,
    )
    assert_eq(model.latent_time, time)
  }
  try
    @minimax_h3_media_encode.AudioEncoder::new(
      samples=0,
      max_scratch_bytes=2000000000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted empty waveform")
  }
}

///|
test "all live SSA regions are disjoint at every launch" {
  let model = @minimax_h3_media_encode.AudioEncoder::new(
    samples=1601,
    max_scratch_bytes=2000000000L,
  )
  let g = model.graph
  for i, node in g.nodes {
    let out_start = g.offsets[node.output]
    let out_end = out_start + g.values[node.output].elements * 4L
    for input in node.inputs {
      if g.values[input].origin == Temporary {
        let start = g.offsets[input]
        let end = start + g.values[input].elements * 4L
        assert_true(out_end <= start || end <= out_start)
      }
    }
    for previous in g.nodes[:i] {
      let id = previous.output
      let mut live = id == g.output
      for next in g.nodes[i:] {
        if next.inputs.contains(id) {
          live = true
        }
      }
      if live {
        let start = g.offsets[id]
        let end = start + g.values[id].elements * 4L
        assert_true(out_end <= start || end <= out_start)
      }
    }
  }
}
