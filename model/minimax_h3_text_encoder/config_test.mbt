///|
fn config_fixture() -> Bytes {
  @utf8.encode(
    (
      #|{"model_type":"qwen3_vl","image_token_id":151655,"video_token_id":151656,"vision_start_token_id":151652,"vision_end_token_id":151653,"text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"head_dim":128,"hidden_act":"silu","hidden_size":5120,"intermediate_size":25600,"max_position_embeddings":262144,"num_attention_heads":64,"num_hidden_layers":64,"num_key_value_heads":8,"rms_norm_eps":1e-6,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"},"rope_theta":5000000,"vocab_size":151936}}
    ),
  )
}

///|
test "checkpoint config selects text epsilon and theta, not denoiser defaults" {
  let c = parse_config(config_fixture(), max_bytes=4096)
  assert_eq(c.theta, 5000000.0)
  assert_eq(c.epsilon, 0.000001)
  assert_eq(c.vocabulary_size, 151936)
  assert_eq(c.plan([1, 2, 3]).rows, 3)
  let t = c.rotary_tables(text_positions(2)[:], max_bytes=1024L)
  assert_eq(t.cosine.length(), 512)
  assert_eq(t.sine.length(), 512)
  for i in 0..<128 {
    assert_eq(t.cosine[i * 2], 0x80)
    assert_eq(t.cosine[i * 2 + 1], 0x3f)
    assert_eq(t.sine[i * 2], 0)
    assert_eq(t.sine[i * 2 + 1], 0)
  }
  for i in 0..<128 {
    assert_eq(t.cosine[256 + i], t.cosine[384 + i])
  }
  try c.rotary_tables(text_positions(2)[:], max_bytes=1023L) catch {
    ByteBudgetExceeded => ()
    _ => fail("wrong budget error")
  } noraise {
    _ => fail("overbudget tables")
  }
}

///|
test "visual placement requires every placeholder exactly once" {
  let c = parse_config(config_fixture(), max_bytes=4096)
  let p = c.visual_placement([7, 151655, 151655, 9], [1, 2], [
    (0, 0, 0),
    (1, 1, 1),
    (1, 1, 2),
    (3, 3, 3),
  ])
  assert_eq(p.token_positions.length(), 2)
  let t = c.rotary_tables(p.rotary_positions[:], max_bytes=4096L)
  assert_eq(t.cosine.length(), 1024)
  for wrong in [[2, 1], [1], [1, 2, 3]] {
    try
      c.visual_placement([7, 151655, 151655, 9], wrong, [
        (0, 0, 0),
        (1, 1, 1),
        (1, 1, 2),
        (3, 3, 3),
      ])
    catch {
      InvalidRequest => ()
      _ => fail("wrong placement error")
    } noraise {
      _ => fail("invalid placement accepted")
    }
  }
}

///|
test "interleaved rotary takes H W every third lane and temporal remainder" {
  let c = parse_config(config_fixture(), max_bytes=4096)
  let t = c.rotary_tables([(0, 7, 11)], max_bytes=512L)
  assert_eq(t.sine[0], 0)
  assert_eq(t.sine[1], 0)
  assert_true(t.sine[2] != 0 || t.sine[3] != 0)
  assert_true(t.sine[4] != 0 || t.sine[5] != 0)
  for lane in [60, 61, 62, 63] {
    assert_eq(t.sine[lane * 2], 0)
    assert_eq(t.sine[lane * 2 + 1], 0)
  }
}

///|
test "configuration rejects unsupported scaling and duplicate fields" {
  let source = @utf8.decode(config_fixture())
  for
    text in [
      source.replace(old="\"default\"", new="\"yarn\""),
      source.replace(
        old="\"hidden_size\":5120",
        new="\"hidden_size\":5120,\"hidden_size\":5120",
      ),
    ] {
    try parse_config(@utf8.encode(text), max_bytes=4096) catch {
      UnsupportedRotary | InvalidDocument => ()
      _ => fail("wrong config error")
    } noraise {
      _ => fail("invalid config accepted")
    }
  }
}
