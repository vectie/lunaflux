///|
async fn fixture() -> Bytes {
  @fs.read_file("model/minicpm_v47/config.fixture.json").binary()
}

///|
async test "selected flagship keeps BF16 hybrid geometry and compressed vision" {
  let spec = @minicpm_v47.Spec::parse(fixture()[:])
  assert_eq(spec.decoder().vocabulary_size(), 248144)
  assert_eq(spec.decoder().full_layer_count(), 10)
  assert_eq(spec.decoder().recurrent_layer_count(), 30)
  assert_eq(spec.decoder().experts().routed, 256)
  assert_eq(spec.decoder().experts().selected, 8)
  assert_eq(spec.vision().layers, 27)
  assert_eq(spec.vision().position_grid_side, 70)
  assert_eq(spec.vision().compression_after_layer, 6)
  assert_true(spec.downsample() is SixteenX)
  assert_eq(spec.markers().image_start, 248078)
}

///|
async test "wrong model precision schedule vocabulary or vision is rejected" {
  let source = @utf8.decode(fixture())
  for
    (old, new) in [
      ("minicpmv4_7\"", "qwen3_5_moe\""),
      ("bfloat16", "float8_e4m3fn"),
      ("248144", "248320"),
      ("\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 9"),
      ("\"linear_attention\"", "\"full_attention\""),
      ("\"hidden_size\": 1152", "\"hidden_size\": 768"),
    ] {
    let mutated = source.replace(old~, new~)
    assert_true(mutated != source)
    assert_true(
      try @minicpm_v47.Spec::parse(@utf8.encode(mutated)[:]) catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
  let with_unknown = source.replace(
    old="\"dtype\":",
    new="\"auto_map\":{},\"dtype\":",
  )
  assert_true(
    try @minicpm_v47.Spec::parse(@utf8.encode(with_unknown)[:]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
