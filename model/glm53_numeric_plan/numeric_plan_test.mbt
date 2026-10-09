///|
fn numeric_full_metadata(
  digest? : String = "8123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn numeric_flash_metadata(
  digest? : String = "9123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(digest),
  )
}

///|
test "full numeric plan binds all logical tensors and exact schedule" {
  let plan = @glm53_numeric_plan.build_full_numeric_plan(
    numeric_full_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(plan.profile(), FullText)
  assert_eq(plan.tensor_count(), 59585)
  assert_eq(plan.bfloat16_tensor_count(), 59509)
  assert_eq(plan.float32_tensor_count(), 76)
  assert_eq(plan.operation_count(), 960)
  assert_eq(plan.tensor(0).unwrap().ordinal(), 0)
  assert_eq(plan.tensor(0).unwrap().role(), BFloat16Parameter)
  assert_eq(plan.operation(0).unwrap().kind(), TokenEmbedding)
  assert_eq(plan.operation(1).unwrap().scope(), DecoderLayer(layer_index=0))
  assert_eq(plan.operation(959).unwrap().kind(), LanguageModelHead)
  assert_eq(plan.numeric_schema().tensors().length(), plan.tensor_count())
  assert_eq(plan.numeric_schema().operations().length(), plan.operation_count())
}

///|
test "Flash numeric plan preserves F32 KDA hyper and routing roles" {
  let plan = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(plan.profile(), FlashMultimodal)
  assert_eq(plan.tensor_count(), 38770)
  assert_eq(plan.bfloat16_tensor_count(), 38479)
  assert_eq(plan.float32_tensor_count(), 291)
  assert_eq(plan.operation_count(), 998)
  let mut saw_routing = false
  let mut saw_kda = false
  let mut saw_hyper = false
  for ordinal in 0..<plan.tensor_count() {
    match plan.tensor(ordinal).unwrap().role() {
      Float32RoutingCorrection => saw_routing = true
      Float32KdaStateParameter => saw_kda = true
      Float32HyperConnectionParameter => saw_hyper = true
      BFloat16Parameter | Fp8Parameter | Fp8BlockScaleMetadata => ()
    }
  }
  assert_true(saw_routing)
  assert_true(saw_kda)
  assert_true(saw_hyper)
  assert_eq(plan.operation(997).unwrap().kind(), LayerNorm)
  assert_eq(plan.operation(997).unwrap().scope(), VisionProjector)
}

///|
test "numeric identity is deterministic and content profile sensitive" {
  let first = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  let second = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(first.binding_digest(), second.binding_digest())
  assert_eq(first.numeric_schema_digest(), second.numeric_schema_digest())
  let changed = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(
      digest="a123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    BFloat16WithFloat32Routing,
  )
  assert_true(first.execution_identity() != changed.execution_identity())
  assert_true(
    first.manifest_layout_digest() != changed.manifest_layout_digest(),
  )
  assert_true(first.binding_digest() != changed.binding_digest())
}

///|
test "foreign profile plan and foreign content fail closed" {
  let full_metadata = numeric_full_metadata()
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  try
    @glm53_numeric_plan.bind_numeric_plan(
      full_manifest,
      @glm53.build_flash_execution_plan(numeric_flash_metadata()),
    )
  catch {
    ProfilePlanMismatch => ()
    _ => fail("unexpected foreign-profile failure")
  } noraise {
    _ => fail("foreign profile admitted")
  }
  let changed = numeric_full_metadata(
    digest="b123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  try
    @glm53_numeric_plan.bind_numeric_plan(
      full_manifest,
      @glm53.build_full_execution_plan(changed),
    )
  catch {
    ManifestIdentityMismatch => ()
    _ => fail("unexpected foreign-content failure")
  } noraise {
    _ => fail("foreign content admitted")
  }
}

///|
test "FP8 block scales and mixed materialization remain typed failures" {
  try
    @glm53_numeric_plan.build_full_numeric_plan(
      numeric_full_metadata(),
      Fp8E4M3Block128Dynamic,
    )
  catch {
    UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128) => ()
    _ => fail("unexpected FP8 failure")
  } noraise {
    _ => fail("unsupported FP8 scale layout admitted")
  }
  try
    @glm53_numeric_plan.build_flash_numeric_plan(
      numeric_flash_metadata(),
      Fp8E4M3Block128Dynamic,
    )
  catch {
    UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128) => ()
    _ => fail("unexpected Flash FP8 failure")
  } noraise {
    _ => fail("unsupported Flash FP8 scale layout admitted")
  }
  let plan = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  try plan.require_physical_materialization() catch {
    MixedDTypeMaterializationUnsupported => ()
    _ => fail("unexpected mixed-dtype failure")
  } noraise {
    _ => fail("mixed-dtype materialization authority granted")
  }
}

///|
test "ordinal access is bounded and semantic execution contracts are exact" {
  let plan = @glm53_numeric_plan.build_flash_numeric_plan(
    numeric_flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_true(plan.tensor(-1) is None)
  assert_true(plan.tensor(plan.tensor_count()) is None)
  assert_true(plan.operation(-1) is None)
  assert_true(plan.operation(plan.operation_count()) is None)
  let mut saw_f32 = false
  let mut saw_mixed_kda = false
  for ordinal in 0..<plan.operation_count() {
    let binding = plan.operation(ordinal).unwrap()
    assert_eq(binding.ordinal(), ordinal)
    if binding.kind() is (KdaF32DecayControl | MoeF32SigmoidAndCorrection) {
      saw_f32 = true
      assert_eq(
        binding.execution().activation_compute(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        binding.execution().tensor_input(),
        @numeric_contract.ComputeDType::f32(),
      )
    }
    if binding.kind() == MoeRouterProjection {
      assert_eq(
        binding.execution().activation_compute(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        binding.execution().tensor_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        binding.execution().output(),
        @numeric_contract.ComputeDType::f32(),
      )
    }
    if binding.kind() == KdaRecurrentUpdate {
      saw_mixed_kda = true
      assert_eq(
        binding.execution().activation_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        binding.execution().activation_compute(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        binding.execution().output(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
    }
  }
  assert_true(saw_f32)
  assert_true(saw_mixed_kda)
}
