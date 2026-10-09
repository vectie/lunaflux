///|
test "official MiniMax H3 variants normalize to exact schema-v1 geometry" {
  let fl2va = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant=Fl2va,
    checkpoint_format=OriginalV1,
  )
  let ref2va = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant=Ref2va,
    checkpoint_format=DiffusersV1,
  )
  assert_eq(fl2va.schema_version(), 1)
  assert_eq(fl2va.variant(), Fl2va)
  assert_eq(ref2va.variant(), Ref2va)
  assert_eq(fl2va.checkpoint_format().architecture(), "MiniMaxH3DiTModel")
  assert_eq(
    ref2va.checkpoint_format().architecture(),
    "MiniMaxH3Transformer3DModel",
  )
  assert_eq(fl2va.hidden_size(), 5376)
  assert_eq(fl2va.attention_width(), 7168)
  assert_eq(fl2va.adaln_output_features(), 96768)
  assert_eq(fl2va.final_adaln_output_features(), 10752)
  assert_eq(fl2va.patch_shape(), [1, 2, 2])
  assert_eq(ref2va.audio_latent_channels(), 32)
  assert_eq(ref2va.rope_frequency_dimension(), 16)
  assert_eq(ref2va.weight_dtype(), BFloat16)
  assert_eq(ref2va.precision_profile(), OfficialF32BoundaryBFloat16Core)
  assert_eq(ref2va.boundary_weight_dtype(), Float32)
}

///|
fn exact_with(
  hidden_size? : Int = 5376,
  schema_version? : Int = 1,
  weight_dtype? : @spec.ModelDType = BFloat16,
  time_embedding_dimension? : Int = 2688,
) -> @minimax_h3_spec.MiniMaxH3TransformerSpec raise @minimax_h3_spec.MiniMaxH3SpecError {
  @minimax_h3_spec.MiniMaxH3TransformerSpec::new_exact(
    variant=Fl2va,
    checkpoint_format=DiffusersV1,
    schema_version~,
    hidden_size~,
    layer_count=50,
    refiner_layer_count=2,
    attention_head_count=56,
    attention_head_dimension=128,
    ffn_dimension=14336,
    visual_latent_channels=24,
    audio_latent_channels=32,
    patch_temporal=1,
    patch_height=2,
    patch_width=2,
    text_dimension=5120,
    frequency_dimension=256,
    time_embedding_hidden_dimension=5376,
    time_embedding_dimension~,
    adaln_output_features=96768,
    final_adaln_output_features=10752,
    rope_frequency_dimension=16,
    rope_theta=10000.0,
    norm_epsilon=0.00001,
    query_key_norm_epsilon=0.00001,
    final_norm_epsilon=0.00001,
    weight_dtype~,
  )
}

///|
test "MiniMax H3 rejects nearby geometry versions and precision" {
  try exact_with(hidden_size=5377) catch {
    InvalidOfficialValue(HiddenSize) => ()
    _ => fail("unexpected MiniMax H3 geometry error")
  } noraise {
    _ => fail("nearby MiniMax H3 geometry was admitted")
  }
  try exact_with(schema_version=2) catch {
    UnsupportedSchemaVersion => ()
    _ => fail("unexpected MiniMax H3 schema error")
  } noraise {
    _ => fail("unknown MiniMax H3 schema was admitted")
  }
  try exact_with(time_embedding_dimension=2687) catch {
    InvalidOfficialValue(TimeEmbeddingDimension) => ()
    _ => fail("unexpected final-AdaLN conditioning width error")
  } noraise {
    _ => fail("nearby final-AdaLN conditioning width was admitted")
  }
  try exact_with(weight_dtype=Float16) catch {
    UnsupportedWeightDType => ()
    _ => fail("unexpected MiniMax H3 dtype error")
  } noraise {
    _ => fail("non-official MiniMax H3 precision was admitted")
  }
}
