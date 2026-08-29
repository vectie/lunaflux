///|
fn i8_llama_test_metadata(
  layer_count? : Int = 2,
) -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count~,
    attention_head_count=4,
    kv_head_count=2,
    context_length=64,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
    ),
  )
}

///|
fn expected_i8_parameter(index : Int, layer_count : Int) -> Bool {
  if index == 0 || index == 9 * layer_count + 1 {
    false
  } else if index == 9 * layer_count + 2 {
    true
  } else {
    let within_layer = (index - 1) % 9
    within_layer == 1 ||
    within_layer == 2 ||
    within_layer == 3 ||
    within_layer == 4 ||
    within_layer == 6 ||
    within_layer == 7 ||
    within_layer == 8
  }
}

///|
fn expected_i8_operation(index : Int, layer_count : Int) -> Bool {
  if index == 9 * layer_count + 2 {
    true
  } else if index <= 0 || index > 9 * layer_count {
    false
  } else {
    let within_layer = (index - 1) % 9
    within_layer == 1 || within_layer == 4 || within_layer == 7
  }
}

///|
fn expected_bf16_execution(
  has_activation : Bool,
  has_tensor : Bool,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=if has_activation {
      @numeric_contract.ComputeDType::bfloat16()
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    activation_compute=if has_activation {
      @numeric_contract.ComputeDType::bfloat16()
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=if has_tensor {
      @numeric_contract.ComputeDType::bfloat16()
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn operation_has_tensor(kind : @plan.OperationKind) -> Bool {
  kind
  is (Embedding
  | RmsNorm
  | QkvProjection
  | OutputProjection
  | GatedMlp
  | LanguageModelHead)
}

///|
fn operation_has_activation(kind : @plan.OperationKind) -> Bool {
  kind != Embedding
}

///|
test "dense Llama I8 policy fails closed for tied embeddings" {
  try @llama.LlamaI8WeightOnlyPolicy::new(tied_embeddings=true) catch {
    TiedEmbeddingsUnsupported => ()
  } noraise {
    _ => fail("tied embedding and language-model-head storage was admitted")
  }
  assert_eq(
    @llama.LlamaI8WeightOnlyPolicy::new(tied_embeddings=false),
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
  )
}

///|
test "dense Llama I8 plan quantizes every and only eligible parameter" {
  let layer_count = 2
  let plan = @llama.build_symmetric_i8_weight_only_v1(
    i8_llama_test_metadata(layer_count~),
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
  )
  let tensors = plan.numeric_binding().tensors()
  let parameter_count = 3 + 9 * layer_count
  let scale_count = 1 + 7 * layer_count
  assert_eq(plan.operations().length(), parameter_count)
  assert_eq(tensors.length(), parameter_count + scale_count)
  let scale_seen = FixedArray::make(scale_count, false)
  let mut observed_i8 = 0
  for index in 0..<parameter_count {
    let tensor = tensors[index]
    assert_eq(tensor.reference().as_int(), index)
    assert_eq(tensor.role(), ParameterTensor)
    if expected_i8_parameter(index, layer_count) {
      observed_i8 += 1
      assert_eq(
        tensor.storage(),
        @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1(),
      )
      guard tensor.scale() is Some(scale) else {
        fail("eligible I8 parameter owns no scale")
      }
      let scale_index = scale.tensor().as_int() - parameter_count
      assert_true(scale_index >= 0 && scale_index < scale_count)
      assert_false(scale_seen[scale_index])
      scale_seen[scale_index] = true
      let scale_tensor = tensors[parameter_count + scale_index]
      assert_eq(scale_tensor.reference(), scale.tensor())
      assert_eq(scale_tensor.role(), ScaleMetadataTensor)
      assert_eq(
        scale_tensor.storage(),
        @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
      )
      assert_eq(scale_tensor.shape().rank(), 1)
      assert_eq(scale_tensor.shape().dimension(0), tensor.shape().dimension(0))
    } else {
      assert_eq(
        tensor.storage().dtype(),
        @numeric_contract.StorageDType::bfloat16(),
      )
      assert_eq(tensor.scale(), None)
    }
  }
  assert_eq(observed_i8, scale_count)
  for seen in scale_seen {
    assert_true(seen)
  }
}

///|
test "dense Llama I8 operation contracts correspond exactly to graph kinds" {
  let layer_count = 2
  let plan = @llama.build_symmetric_i8_weight_only_v1(
    i8_llama_test_metadata(layer_count~),
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
  )
  let bindings = plan.numeric_binding().operations()
  assert_eq(bindings.length(), plan.operations().length())
  let i8_execution = @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
  let mut observed_i8 = 0
  for index, operation in plan.operations() {
    let binding = bindings[index]
    assert_eq(binding.operation().as_int(), index)
    if expected_i8_operation(index, layer_count) {
      observed_i8 += 1
      assert_eq(binding.execution(), i8_execution)
      assert_true(
        operation.kind()
        is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead),
      )
    } else {
      assert_eq(
        binding.execution(),
        expected_bf16_execution(
          operation_has_activation(operation.kind()),
          operation_has_tensor(operation.kind()),
        ),
      )
    }
  }
  assert_eq(observed_i8, 1 + 3 * layer_count)
}

///|
test "dense Llama I8 identities are deterministic and mode-specific" {
  let metadata = i8_llama_test_metadata()
  let policy = @llama.LlamaI8WeightOnlyPolicy::untied_v1()
  let first = @llama.build_symmetric_i8_weight_only_v1(metadata, policy)
  let repeated = @llama.build_symmetric_i8_weight_only_v1(metadata, policy)
  let bf16 = @llama.build(metadata)
  let paged = @llama.build_paged_symmetric_i8_weight_only_v1(metadata, policy)
  assert_eq(first, repeated)
  assert_eq(first.identity().content(), metadata.content_digest())
  assert_eq(first.identity(), repeated.identity())
  assert_true(first.identity().plan() != bf16.identity().plan())
  assert_true(first.identity().plan() != paged.identity().plan())
  assert_true(
    first.numeric_binding().digest() != bf16.numeric_binding().digest(),
  )
  inspect(
    first.identity().plan().as_hex(),
    content="c83f4ef9af74a24d710db0eb3ba6032b5750b7f8c52134afb755a84bba3e9e86",
  )
  inspect(
    first.numeric_binding().digest().as_hex(),
    content="386baef801ca914de2c27c2b82018f4694045837388d91fbcaa926ccf519ac4c",
  )
}

///|
test "dense Llama I8 preserves the exact 454-layer construction ceiling" {
  let maximum = @llama.build_symmetric_i8_weight_only_v1(
    i8_llama_test_metadata(layer_count=454),
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
  )
  assert_eq(maximum.operations().length(), 4089)
  assert_eq(maximum.numeric_binding().tensors().length(), 7268)
  let mut total_value_inputs = 0
  for operation in maximum.operations() {
    total_value_inputs += operation.inputs().length()
  }
  assert_eq(total_value_inputs, 9086)
  try i8_llama_test_metadata(layer_count=455) catch {
    @spec.InvalidDimension(LayerCount) => ()
    _ => fail("unexpected layer-ceiling failure")
  } noraise {
    _ => fail("455-layer metadata bypassed the preallocation ceiling")
  }
}
