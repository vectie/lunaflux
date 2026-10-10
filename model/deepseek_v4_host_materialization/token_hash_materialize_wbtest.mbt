///|
fn token_hash_test_write_i64(
  arena : FixedArray[Byte],
  offset : Int,
  value : Int,
) -> Unit {
  let raw = value.to_int64().reinterpret_as_uint64()
  for shift in 0..<8 {
    arena[offset + shift] = ((raw >> (shift * 8)) & 0xffUL).to_byte()
  }
}

///|
fn token_hash_test_weights() -> (
  @deepseek_v4.DeepSeekV4SemanticPlan,
  DeepSeekV4HostWeights,
) {
  let spec = try! @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    try! @spec.ContentDigest::from_sha256(String::make(64, 'a')),
  )
  let plan = @deepseek_v4.build(metadata)
  let execution = plan.execution_plan()
  let identity = try! @spec.ModelIdentity::new(
    execution.content_digest(),
    execution.plan_digest(),
  )
  let elements = spec.vocabulary_size() * spec.experts_per_token()
  let table_bytes = elements * 8
  let arena = FixedArray::make(table_bytes * spec.hash_layer_count(), b'\x00')
  let tensors : Array[DeepSeekV4HostTensor] = []
  for layer in 0..<spec.hash_layer_count() {
    let base = layer * table_bytes
    for element in 0..<elements {
      token_hash_test_write_i64(
        arena,
        base + element * 8,
        element % spec.experts_per_token(),
      )
    }
    let source = try! @deepseek_v4_weights.DeepSeekV4SafetensorsTensor::new(
      name="layers.\{layer}.ffn.gate.tid2eid",
      dtype="I64",
      shape=[
        spec.vocabulary_size().to_int64(),
        spec.experts_per_token().to_int64(),
      ],
      shard=0,
      data_start=0L,
      data_end=table_bytes.to_int64(),
    )
    let binding = try! @deepseek_v4_weights.validate_tensor_semantic(
      spec, source,
    )
    tensors.push({
      binding,
      arena_ordinal: 0,
      arena_offset: base,
      byte_count: table_bytes,
      quantized_parameter_index: None,
    })
  }
  let expected = DeepSeekV4HostExpectedDigests::new(
    identity~,
    logical_manifest_digest=execution.plan_digest(),
    numeric_schema_digest=execution.plan_digest(),
    numeric_binding_digest=execution.plan_digest(),
  )
  let manifest_digest : DeepSeekV4HostManifestDigest = {
    hex: String::make(64, 'b'),
  }
  let weights : DeepSeekV4HostWeights = {
    identity,
    expected_digests: expected,
    manifest_digest,
    tensors: ReadOnlyArray::from_array(tensors),
    arenas: [Some(arena)],
    total_tensor_bytes: arena.length().to_int64(),
    total_arena_bytes: arena.length().to_int64(),
    released: false,
  }
  (plan, weights)
}

///|
test "official DeepSeek hash tables checked-narrow without an I64 staging copy" {
  let (plan, weights) = token_hash_test_weights()
  let foreign_spec = try! @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let foreign_plan = @deepseek_v4.build(
    @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
      foreign_spec,
      try! @spec.ContentDigest::from_sha256(String::make(64, 'c')),
    ),
  )
  try
    weights.materialize_token_hash_tables(
      plan=foreign_plan,
      limits=try! DeepSeekV4TokenHashMaterializationLimits::new(
        maximum_tables=3,
        maximum_total_i32_bytes=10000000L,
      ),
    )
  catch {
    TokenHashIdentityMismatch => ()
    _ => fail("unexpected token-hash identity rejection")
  } noraise {
    _ => fail("foreign plan identity was admitted")
  }
  try
    weights.materialize_token_hash_tables(
      plan~,
      limits=try! DeepSeekV4TokenHashMaterializationLimits::new(
        maximum_tables=3,
        maximum_total_i32_bytes=9000000L,
      ),
    )
  catch {
    TokenHashOutputBudgetExceeded => ()
    _ => fail("unexpected token-hash budget rejection")
  } noraise {
    _ => fail("undersized token-hash budget was admitted")
  }
  let tables = weights.materialize_token_hash_tables(
    plan~,
    limits=try! DeepSeekV4TokenHashMaterializationLimits::new(
      maximum_tables=3,
      maximum_total_i32_bytes=10000000L,
    ),
  )
  assert_eq(tables.identity(), weights.identity())
  assert_eq(tables.source_manifest_digest(), weights.manifest_digest())
  assert_eq(tables.table_count(), 3)
  let expected_bytes = 129280 * 6 * 4
  for layer in 0..<3 {
    assert_eq(tables.layer_index(layer), layer)
    assert_eq(tables.source_tensor_index(layer), layer)
    assert_eq(tables.table_byte_count(layer), expected_bytes)
    assert_eq(tables.table_source_label(layer), tables.source_manifest_digest())
  }
  let observed = Ref(false)
  let consumer = try! @segmented_host_borrow.SegmentedHostArenaConsumer::new(
    arena_ordinal=0,
    expected_length=expected_bytes,
    consume=arena => {
      observed.val = arena[0] == 0 &&
        arena[4] == 1 &&
        arena[8] == 2 &&
        arena[12] == 3 &&
        arena[16] == 4 &&
        arena[20] == 5
      true
    },
  )
  tables.provide_table_arena(0, consumer)
  consumer.revoke()
  assert_true(consumer.was_accepted())
  assert_true(observed.val)
  assert_true(tables.release())
  assert_false(tables.release())
  assert_true(tables.is_released())
  assert_true(weights.release())
}
