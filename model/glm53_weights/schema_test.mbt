///|
fn full_schema_fixture() -> @glm53_weights.Glm53WeightSchema {
  let metadata = @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  @glm53_weights.full_schema(metadata, BFloat16WithFloat32Routing)
}

///|
fn flash_schema_fixture() -> @glm53_weights.Glm53WeightSchema {
  let metadata = @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  @glm53_weights.flash_schema(metadata, Fp8E4M3Block128Dynamic)
}

///|
test "full weight schema distinguishes dense and MoE layers" {
  let schema = full_schema_fixture()
  assert_eq(schema.global_shape(TokenEmbedding), [154880L, 6144L])
  assert_eq(schema.text_layer_shape(0, DenseGate), [12288L, 6144L])
  assert_eq(schema.text_layer_shape(3, RoutedGateUp), [256L, 4096L, 6144L])
  assert_eq(schema.text_layer_shape(3, QueryB), [16384L, 2048L])
  try schema.text_layer_shape(3, IndexQuery) catch {
    TensorRoleNotPresent(IndexQuery) => ()
    _ => fail("unexpected shared-indexer role error")
  } noraise {
    _ => fail("indexer tensor admitted in a shared DSA layer")
  }
  try schema.text_layer_shape(0, RoutedDown) catch {
    TensorRoleNotPresent(RoutedDown) => ()
    _ => fail("unexpected full role error")
  } noraise {
    _ => fail("MoE tensor admitted in a dense full layer")
  }
}

///|
test "Flash schema distinguishes KDA and DSA tensors" {
  let schema = flash_schema_fixture()
  assert_eq(schema.fp8_block_rows(), Some(128))
  assert_eq(schema.fp8_block_columns(), Some(128))
  assert_eq(schema.text_layer_shape(0, LinearQuery), [8192L, 4096L])
  assert_eq(schema.text_layer_shape(0, LinearConvolution), [24576L, 1L, 4L])
  assert_eq(schema.text_layer_shape(0, HyperAttentionFunction), [24L, 16384L])
  assert_eq(schema.text_layer_shape(3, QueryB), [16384L, 1536L])
  assert_eq(schema.text_layer_shape(3, IndexPoolPosition), [4L, 128L])
  try schema.text_layer_shape(3, LinearOutput) catch {
    TensorRoleNotPresent(LinearOutput) => ()
    _ => fail("unexpected Flash role error")
  } noraise {
    _ => fail("KDA tensor admitted in a DSA Flash layer")
  }
}

///|
test "weight schemas reject hostile layer indices" {
  try flash_schema_fixture().text_layer_shape(45, InputNorm) catch {
    InvalidLayerIndex => ()
    _ => fail("unexpected layer-bound error")
  } noraise {
    _ => fail("out-of-range Flash weight layer was admitted")
  }
}
