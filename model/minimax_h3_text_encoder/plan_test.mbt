///|
test "extractor excludes final norm and upper fourteen checkpoint layers" {
  let p = TextEncoderPlan::new(rows=3, epsilon=0.000001)
  assert_eq(p.selected_hidden_state(), 50)
  assert_eq(p.output_bytes(), 30720L)
  let w = layer_weights(49)
  assert_eq(w.regions.length(), 11)
  assert_eq(
    w.regions[8].name,
    "model.language_model.layers.49.mlp.up_proj.weight",
  )
  assert_eq(w.regions[9].offset, w.regions[8].offset + w.regions[8].bytes)
  try layer_weights(50) catch {
    InvalidLayer => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("upper layer admitted")
  }
  let r = p.workspace_regions()
  for i in 1..<r.length() {
    assert_eq(r[i].0, r[i - 1].0 + r[i - 1].1)
  }
}

///|
test "text-only encoder never silently discards visual conditioning" {
  let p = TextEncoderPlan::new(rows=2, epsilon=0.000001)
  p.validate_text_tokens([1, 2], image_token=151655, video_token=151656)
  try
    p.validate_text_tokens([1, 151655], image_token=151655, video_token=151656)
  catch {
    VisualTokensRequireEncoder => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("visual token admitted")
  }
  try
    p.validate_text_tokens([1, -1], image_token=151655, video_token=151656)
  catch {
    InvalidToken => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("invalid token admitted")
  }
}

///|
test "all fifty layer packings cover eleven disjoint BF16 tensors" {
  for layer in 0..<50 {
    let w = layer_weights(layer)
    let mut end = 0L
    for r in w.regions {
      assert_eq(r.offset, end)
      assert_true(r.bytes > 0L && r.bytes % 2L == 0L)
      end += r.bytes
    }
    assert_eq(end, w.byte_count)
    assert_eq(w.regions[1].bytes, 8192L * 5120L * 2L)
    assert_eq(w.regions[2].bytes, 1024L * 5120L * 2L)
    assert_eq(w.regions[10].bytes, 5120L * 25600L * 2L)
  }
  for rows in [0, -1, 262145] {
    try TextEncoderPlan::new(rows~, epsilon=0.000001) catch {
      InvalidRows => ()
      _ => fail("wrong rows error")
    } noraise {
      _ => fail("invalid rows admitted")
    }
  }
}
