///|
fn canonical_tag_shape_hash(shape : OperationShape) -> String {
  let writer = ModelPlanCanonicalWriter::new_hashing(128)
  try! encode_operation_shape(writer, shape)
  writer.finish_hex()
}

///|
fn canonical_tag_value_hash(value : ValueRef) -> String {
  let writer = ModelPlanCanonicalWriter::new_hashing(16)
  try! encode_value_ref(writer, value)
  writer.finish_hex()
}

///|
fn canonical_tag_read_u64(bytes : Bytes, offset : Int) -> UInt64 {
  let mut value = 0UL
  for index in 0..<8 {
    value = value | (bytes[offset + index].to_uint64() << (index * 8))
  }
  value
}

///|
test "canonical operation and runtime tag tables are pinned" {
  assert_eq(
    [
      execution_mode_tag(StatelessFullContext),
      execution_mode_tag(PagedKeyValue),
    ],
    [0x01, 0x02],
  )
  assert_eq(
    [
      model_dtype_tag(BFloat16),
      model_dtype_tag(Float16),
      model_dtype_tag(Float32),
    ],
    [0x01, 0x02, 0x03],
  )
  assert_eq(
    [
      operation_kind_tag(Embedding),
      operation_kind_tag(RmsNorm),
      operation_kind_tag(QkvProjection),
      operation_kind_tag(RotaryEmbedding),
      operation_kind_tag(CausalAttention),
      operation_kind_tag(OutputProjection),
      operation_kind_tag(ResidualAdd),
      operation_kind_tag(GatedMlp),
      operation_kind_tag(LanguageModelHead),
    ],
    [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09],
  )
  assert_eq(
    [
      runtime_input_tag(StepCounts),
      runtime_input_tag(QueryPositions),
      runtime_input_tag(QueryRowOffsets),
      runtime_input_tag(SequenceLengths),
      runtime_input_tag(PageTableOffsets),
      runtime_input_tag(PageTablePageIndices),
    ],
    [0x01, 0x02, 0x03, 0x04, 0x05, 0x06],
  )
  assert_eq(LayerMajorSplitKeyValueV1.stable_version(), 1)
  assert_eq(kv_cache_layout_tag(LayerMajorSplitKeyValueV1), 0x01)
  assert_eq(
    [
      value_ref_tag(ModelInput(TokenIds)),
      value_ref_tag(Tensor({ value: 0 })),
      value_ref_tag(Activation({ value: 0 })),
    ],
    [0x01, 0x02, 0x03],
  )
  assert_eq(
    [
      operation_shape_tag(EmbeddingShape(vocabulary_size=8, width=4)),
      operation_shape_tag(NormalizationShape(width=4, epsilon=0.000001)),
      operation_shape_tag(ProjectionShape(input_width=4, output_width=8)),
      operation_shape_tag(
        RotaryShape(
          query_heads=2,
          kv_heads=1,
          head_dimension=2,
          rope_theta=10000.0,
        ),
      ),
      operation_shape_tag(
        AttentionShape(query_heads=2, kv_heads=1, head_dimension=2),
      ),
      operation_shape_tag(ElementwiseShape(width=4)),
      operation_shape_tag(GatedMlpShape(hidden_size=4, intermediate_size=8)),
    ],
    [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
  )
  assert_eq(
    [optional_kv_layer_tag(None), optional_kv_layer_tag(Some({ value: 0 }))],
    [0x00, 0x01],
  )
  assert_eq(
    [
      persistent_state_tag(NoPersistentState),
      persistent_state_tag(KvCacheReadWrite({ value: 0 })),
    ],
    [0x00, 0x01],
  )
}

///|
test "all canonical operation-shape tags and payload forms are distinct" {
  let hashes = [
    canonical_tag_shape_hash(EmbeddingShape(vocabulary_size=8, width=4)),
    canonical_tag_shape_hash(NormalizationShape(width=4, epsilon=0.000001)),
    canonical_tag_shape_hash(ProjectionShape(input_width=4, output_width=8)),
    canonical_tag_shape_hash(
      RotaryShape(
        query_heads=2,
        kv_heads=1,
        head_dimension=2,
        rope_theta=10000.0,
      ),
    ),
    canonical_tag_shape_hash(
      AttentionShape(query_heads=2, kv_heads=1, head_dimension=2),
    ),
    canonical_tag_shape_hash(ElementwiseShape(width=4)),
    canonical_tag_shape_hash(GatedMlpShape(hidden_size=4, intermediate_size=8)),
  ]
  for index, hash in hashes {
    for prior in 0..<index {
      assert_false(hash == hashes[prior])
    }
  }
}

///|
test "value variants and count framing affect canonical operation bytes" {
  let model_input = canonical_tag_value_hash(ModelInput(TokenIds))
  let tensor = canonical_tag_value_hash(Tensor({ value: 0 }))
  let activation = canonical_tag_value_hash(Activation({ value: 0 }))
  assert_false(model_input == tensor)
  assert_false(model_input == activation)
  assert_false(tensor == activation)
  let baseline = canonical_wb_hash()
  let operation = canonical_wb_operation()
  let count_mutations = [
    canonical_wb_clone_operation(operation, inputs=[
      ModelInput(TokenIds),
      Tensor({ value: 0 }),
      Activation({ value: 0 }),
    ]),
    canonical_wb_clone_operation(operation, outputs=[{ value: 0 }, { value: 1 }]),
    canonical_wb_clone_operation(operation, runtime_inputs=[
      StepCounts,
      QueryPositions,
    ]),
  ]
  for index, mutation in count_mutations {
    assert_canonical_mutation(
      baseline,
      canonical_wb_hash(operations=[mutation]),
      "operation-count-framing-\{index}",
    )
  }
}

///|
test "minimal oracle pins exact count offsets and encoded length" {
  let bytes = canonical_wb_manual_minimal_bytes()
  assert_eq(bytes.length(), 289)
  assert_eq(bytes[126], 0x05)
  assert_eq(canonical_tag_read_u64(bytes, 127), 1UL)
  assert_eq(bytes[135], 0x06)
  assert_eq(canonical_tag_read_u64(bytes, 145), 2UL)
  assert_eq(canonical_tag_read_u64(bytes, 171), 1UL)
  assert_eq(canonical_tag_read_u64(bytes, 221), 0UL)
  assert_eq(bytes[230], 0x07)
  assert_eq(canonical_tag_read_u64(bytes, 231), 1UL)
  assert_eq(bytes[256], 0x09)
}
