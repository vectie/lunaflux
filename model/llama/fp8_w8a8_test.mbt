///|
fn fp8_llama_test_metadata(
  layer_count? : Int = 2,
) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count~,
      attention_head_count=4,
      kv_head_count=2,
      context_length=32,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn expected_fp8_parameter(index : Int, layer_count : Int) -> Bool {
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
fn expected_fp8_operation(index : Int, layer_count : Int) -> Bool {
  index == 9 * layer_count + 2 ||
  (
    index > 0 &&
    index <= 9 * layer_count &&
    ((index - 1) % 9 == 1 || (index - 1) % 9 == 4 || (index - 1) % 9 == 7)
  )
}

///|
fn assert_fp8_shape(
  tensor : @plan.PlanTensor,
  dimensions : ArrayView[Int64],
) -> Unit raise {
  assert_eq(tensor.shape().rank(), dimensions.length())
  for axis, dimension in dimensions {
    assert_eq(tensor.shape().dimension(axis), Some(dimension))
  }
}

///|
test "dense Llama FP8 policy rejects tied embeddings" {
  try @llama.LlamaFp8W8A8Policy::new(tied_embeddings=true) catch {
    TiedFp8EmbeddingsUnsupported => ()
  } noraise {
    _ => fail("tied BF16 embedding and FP8 head were admitted")
  }
  assert_eq(
    @llama.LlamaFp8W8A8Policy::new(tied_embeddings=false),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
}

///|
test "dense Llama FP8 schema binds every exact parameter shape and scale" {
  let layer_count = 2
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v1(
    fp8_llama_test_metadata(layer_count~),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
  let tensors = plan.numeric_binding().tensors()
  let parameter_count = 3 + 9 * layer_count
  let scale_count = 1 + 7 * layer_count
  let fp8_storage = try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::f8_e4m3_finite(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::scaled_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::per_tensor(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
  )
  assert_eq(plan.operations().length(), parameter_count)
  assert_eq(tensors.length(), parameter_count + scale_count)
  assert_fp8_shape(tensors[0], [32L, 8L])
  for layer in 0..<layer_count {
    let base = 1 + 9 * layer
    assert_fp8_shape(tensors[base], [8L])
    assert_fp8_shape(tensors[base + 1], [8L, 8L])
    assert_fp8_shape(tensors[base + 2], [4L, 8L])
    assert_fp8_shape(tensors[base + 3], [4L, 8L])
    assert_fp8_shape(tensors[base + 4], [8L, 8L])
    assert_fp8_shape(tensors[base + 5], [8L])
    assert_fp8_shape(tensors[base + 6], [16L, 8L])
    assert_fp8_shape(tensors[base + 7], [16L, 8L])
    assert_fp8_shape(tensors[base + 8], [8L, 16L])
  }
  assert_fp8_shape(tensors[parameter_count - 2], [8L])
  assert_fp8_shape(tensors[parameter_count - 1], [32L, 8L])
  let scale_seen = FixedArray::make(scale_count, false)
  let mut fp8_count = 0
  for index in 0..<parameter_count {
    let tensor = tensors[index]
    assert_eq(tensor.reference().as_int(), index)
    assert_eq(tensor.role(), ParameterTensor)
    if expected_fp8_parameter(index, layer_count) {
      fp8_count += 1
      assert_eq(tensor.storage(), fp8_storage)
      guard tensor.scale() is Some(scale) else {
        fail("eligible FP8 parameter owns no scale")
      }
      let scale_index = scale.tensor().as_int() - parameter_count
      assert_true(scale_index >= 0 && scale_index < scale_count)
      assert_false(scale_seen[scale_index])
      scale_seen[scale_index] = true
    } else {
      assert_eq(
        tensor.storage().dtype(),
        @numeric_contract.StorageDType::bfloat16(),
      )
      assert_eq(tensor.scale(), None)
    }
  }
  assert_eq(fp8_count, scale_count)
  for index in 0..<scale_count {
    let scale = tensors[parameter_count + index]
    assert_eq(scale.role(), ScaleMetadataTensor)
    assert_eq(scale.storage().dtype(), @numeric_contract.StorageDType::f32())
    assert_eq(scale.shape().rank(), 0)
    assert_true(scale_seen[index])
  }
}

///|
test "dense Llama FP8 execution is homogeneous only across projections" {
  let layer_count = 2
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v1(
    fp8_llama_test_metadata(layer_count~),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
  let fp8 = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  let mut observed = 0
  for index, binding in plan.numeric_binding().operations() {
    assert_eq(binding.operation().as_int(), index)
    if expected_fp8_operation(index, layer_count) {
      observed += 1
      assert_eq(binding.execution(), fp8)
      assert_true(
        plan.operations()[index].kind()
        is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead),
      )
    } else {
      assert_true(binding.execution() != fp8)
    }
  }
  assert_eq(observed, 1 + 3 * layer_count)
}

///|
test "dense Llama FP8 plan identity is deterministic and execution-mode bound" {
  let metadata = fp8_llama_test_metadata()
  let policy = @llama.LlamaFp8W8A8Policy::untied_v1()
  let first = @llama.build_finite_e4m3_fp8_w8a8_v1(metadata, policy)
  let repeated = @llama.build_finite_e4m3_fp8_w8a8_v1(metadata, policy)
  let bf16 = @llama.build(metadata)
  let paged = @llama.build_paged_finite_e4m3_fp8_w8a8_v1(metadata, policy)
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
    content="83393dddd3800f45170b6a9343c1b830c0306e2466c6dfefb153ae3b46a8f49d",
  )
  inspect(
    first.numeric_binding().digest().as_hex(),
    content="5b595e6132e79caba501c0974ad5002749212324097292d09a66f440160dee48",
  )
}

///|
test "dense Llama FP8 preserves the exact 454-layer construction ceiling" {
  let maximum = @llama.build_finite_e4m3_fp8_w8a8_v1(
    fp8_llama_test_metadata(layer_count=454),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
  assert_eq(maximum.operations().length(), 4089)
  assert_eq(maximum.numeric_binding().tensors().length(), 7268)
  let mut total_value_inputs = 0
  for operation in maximum.operations() {
    total_value_inputs += operation.inputs().length()
  }
  assert_eq(total_value_inputs, 9086)
  try fp8_llama_test_metadata(layer_count=455) catch {
    @spec.InvalidDimension(LayerCount) => ()
    _ => fail("unexpected FP8 layer-ceiling failure")
  } noraise {
    _ => fail("455-layer metadata bypassed the FP8 construction ceiling")
  }
}
