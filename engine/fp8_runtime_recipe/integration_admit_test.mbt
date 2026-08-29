///|
fn append_fixture_u64(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn fixture_dtype(tensor : @numeric_contract.NumericTensorSchema) -> String {
  let dtype = tensor.storage().dtype()
  if dtype == @numeric_contract.StorageDType::bfloat16() {
    "BF16"
  } else if dtype == @numeric_contract.StorageDType::f8_e4m3_finite() {
    "F8_E4M3"
  } else {
    "F32"
  }
}

///|
fn fixture_shape(tensor : @numeric_contract.NumericTensorSchema) -> String {
  let dimensions : Array[String] = []
  for dimension in tensor.shape().dimensions() {
    dimensions.push(dimension.to_string())
  }
  "[\{dimensions.join(",")}]"
}

///|
fn recipe_numeric_file(
  plan : @plan.ModelPlan,
  fp8_code? : Byte = b'\x00',
) -> Bytes {
  let tensors = plan.numeric_binding().schema().tensors()
  let header = StringBuilder()
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan.identity().plan().as_hex()}\",\"numeric_schema_sha256\":\"\{plan.numeric_binding().digest().as_hex()}\"}",
  )
  let mut offset = 0L
  for ordinal, tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    )
    header.write_string(
      ",\"lunaflux.tensor.v1.\{ordinal}\":{\"dtype\":\"\{fixture_dtype(tensor)}\",\"shape\":\{fixture_shape(tensor)},\"data_offsets\":[\{offset},\{offset + length}]}",
    )
    offset += length
  }
  header.write_string("}")
  let encoded = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  append_fixture_u64(output, encoded.length().to_uint64())
  for byte in encoded {
    output.push(byte)
  }
  for tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    ).to_int()
    if tensor.storage().dtype() == @numeric_contract.StorageDType::f32() {
      for _ in 0..<(length / 4) {
        output.push(b'\x00')
        output.push(b'\x00')
        output.push(b'\x80')
        output.push(b'\x3f')
      }
    } else {
      let fill = if tensor.storage().dtype() ==
        @numeric_contract.StorageDType::f8_e4m3_finite() {
        fp8_code
      } else {
        b'\x00'
      }
      for _ in 0..<length {
        output.push(fill)
      }
    }
  }
  Bytes::from_array(output)
}

///|
fn inspect_recipe_file(
  root : @approved_fs.ApprovedRoot,
  plan : @plan.ModelPlan,
  name : String,
  file : BytesView,
  alignment? : Int64 = 16L,
) -> @numeric_weight_file.NumericWeightFileInspection {
  try! @numeric_weight_file.inspect_file(
    root,
    plan,
    @approved_fs.ApprovedRelativeLocator::new(name),
    expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(file)),
    ),
    limits=@numeric_weight_file.NumericWeightFileLimits::new(
      max_file_bytes=1048576L,
      max_header_bytes=262144,
      transfer_chunk_bytes=37,
      max_tensor_count=64,
    ),
    alignment_bytes=alignment,
    max_arena_bytes=1048576L,
  )
}

///|
struct RuntimeRecipeInputs {
  launches : Array[@fp8_launch_abi.Fp8DynamicLaunchAbi]
  weights : Array[Fp8LaunchWeightRegionClaim]
}

///|
fn layout_region(
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  tensor : @plan.TensorRef,
  offset_delta? : Int64 = 0L,
  length_delta? : Int64 = 0L,
  alignment? : Int64 = 16L,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  let layout = inspection.layout()
  let exact = layout.region(tensor).unwrap()
  try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      inspection.artifact_digest().as_hex(),
    ),
    storage_bytes=layout.arena_bytes(),
    offset=exact.offset_bytes() + offset_delta,
    byte_length=exact.byte_length() + length_delta,
    alignment~,
  )
}

///|
fn runtime_inputs(
  fixture : RuntimeRecipeFixture,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  workspace? : String = "2222222222222222222222222222222222222222222222222222222222222222",
  workspace_bytes? : Int64 = 64L,
  weight_offset_delta? : Int64 = 0L,
  weight_length_delta? : Int64 = 0L,
  weight_alignment? : Int64 = 16L,
  scale_offset_delta? : Int64 = 0L,
  scale_alignment? : Int64 = 16L,
  claim_operation_delta? : Int = 0,
) -> RuntimeRecipeInputs {
  let launches : Array[@fp8_launch_abi.Fp8DynamicLaunchAbi] = []
  let claims : Array[Fp8LaunchWeightRegionClaim] = []
  let entries = fixture.startup.kernel_artifacts().manifest().entries()
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  for index, entry in entries {
    let workspace_region = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(workspace),
      storage_bytes=workspace_bytes,
      offset=0L,
      byte_length=4L,
      alignment=4L,
    )
    let scales : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    let operands : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    for binding_index, binding in entry.weights() {
      let mutate = index == 0 && binding_index == 0
      scales.push(
        layout_region(
          inspection,
          binding.scale().tensor(),
          offset_delta=if mutate { scale_offset_delta } else { 0L },
          alignment=if mutate { scale_alignment } else { 16L },
        ),
      )
      operands.push(
        layout_region(
          inspection,
          binding.tensor(),
          offset_delta=if mutate { weight_offset_delta } else { 0L },
          length_delta=if mutate { weight_length_delta } else { 0L },
          alignment=if mutate { weight_alignment } else { 16L },
        ),
      )
    }
    launches.push(
      try! @fp8_launch_abi.admit_dynamic_activation_scale_launch(
        fixture.plan,
        fixture.startup.kernel_artifacts(),
        Stateless(fixture.contracts),
        operation=entry.operation(),
        shape=@fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
          profile~,
          @device_profile.ExactExecutionShape::full_prefill(
            batch_rows=1,
            sequence_tokens=2,
          ),
        ),
        weight_scale_regions=scales,
        activation_scale_workspace=workspace_region,
        kernel_workspace=None,
      ),
    )
    claims.push(
      try! Fp8LaunchWeightRegionClaim::new(
        profile~,
        operation=try! @plan.OperationId::new(
          entry.operation().as_int() + claim_operation_delta,
        ),
        regions=operands,
      ),
    )
  }
  { launches, weights: claims }
}

///|
fn recipe_limits(canonical? : Int = 65536) -> Fp8RuntimeRecipeLimits {
  try! Fp8RuntimeRecipeLimits::new(
    max_launches=8,
    max_workspace_bytes=1024L,
    max_canonical_bytes=canonical,
  )
}

///|
fn expect_recipe_rule(
  body : () -> Fp8RuntimeRecipe raise Fp8RuntimeRecipeError,
  expected : Fp8RuntimeRecipeRule,
) -> Unit raise {
  try body() catch {
    InvalidFp8RuntimeRecipe(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("hostile FP8 runtime recipe was admitted")
  }
}

///|
fn expect_scale_length_rejected_before_recipe(
  fixture : RuntimeRecipeFixture,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
) -> Unit raise {
  let entry = fixture.startup.kernel_artifacts().manifest().entries()[0]
  let binding = entry.weights()[0]
  let exact = inspection.layout().region(binding.scale().tensor()).unwrap()
  let invalid_scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      inspection.artifact_digest().as_hex(),
    ),
    storage_bytes=inspection.layout().arena_bytes(),
    offset=exact.offset_bytes() - inspection.layout().alignment_bytes(),
    byte_length=8L,
    alignment=inspection.layout().alignment_bytes(),
  )
  let workspace = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      "5555555555555555555555555555555555555555555555555555555555555555",
    ),
    storage_bytes=64L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  try
    @fp8_launch_abi.admit_dynamic_activation_scale_launch(
      fixture.plan,
      fixture.startup.kernel_artifacts(),
      Stateless(fixture.contracts),
      operation=entry.operation(),
      shape=@fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
        profile~,
        @device_profile.ExactExecutionShape::full_prefill(
          batch_rows=1,
          sequence_tokens=2,
        ),
      ),
      weight_scale_regions=[invalid_scale],
      activation_scale_workspace=workspace,
      kernel_workspace=None,
    )
  catch {
    @fp8_launch_abi.InvalidFp8LaunchAbi(WorkspaceContract) => ()
    _ => fail("unexpected scale-length substitution failure")
  } noraise {
    _ => fail("invalid scale length reached FP8 runtime recipe admission")
  }
}

///|
async test "public admission joins exact evidence and rejects substitutions" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-runtime-recipe")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = runtime_recipe_fixture()
    let file = recipe_numeric_file(fixture.plan)
    let changed = recipe_numeric_file(fixture.plan, fp8_code=b'\x01')
    @fs.write_file("\{directory}/weights.bin", file, create_mode=CreateNew)
    @fs.write_file("\{directory}/changed.bin", changed, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspection = inspect_recipe_file(
      root,
      fixture.plan,
      "weights.bin",
      file,
    )
    let wide_inspection = inspect_recipe_file(
      root,
      fixture.plan,
      "weights.bin",
      file,
      alignment=32L,
    )
    let changed_inspection = inspect_recipe_file(
      root,
      fixture.plan,
      "changed.bin",
      changed,
    )
    let source = runtime_inputs(fixture, inspection)
    let admitted = admit(
      fixture.startup,
      inspection,
      source.launches,
      source.weights,
      recipe_limits(),
    )
    let repeated_inputs = runtime_inputs(fixture, inspection)
    let repeated = admit(
      fixture.startup,
      inspection,
      repeated_inputs.launches,
      repeated_inputs.weights,
      recipe_limits(),
    )
    assert_eq(admitted.digest(), repeated.digest())
    assert_eq(admitted.canonical_bytes(), repeated.canonical_bytes())
    let original_digest = admitted.digest()
    source.launches.push(source.launches[0])
    source.weights.push(source.weights[0])
    assert_eq(admitted.launches().length(), 1)
    assert_eq(admitted.weight_bindings().length(), 1)
    assert_eq(admitted.digest(), original_digest)

    let alternate = runtime_inputs(
      fixture,
      inspection,
      workspace="3333333333333333333333333333333333333333333333333333333333333333",
    )
    let alternate_admitted = admit(
      fixture.startup,
      inspection,
      alternate.launches,
      alternate.weights,
      recipe_limits(),
    )
    assert_true(alternate_admitted.digest() != admitted.digest())
    let wide_inputs = runtime_inputs(
      fixture,
      wide_inspection,
      weight_alignment=32L,
      scale_alignment=32L,
    )
    let wide_admitted = admit(
      fixture.startup,
      wide_inspection,
      wide_inputs.launches,
      wide_inputs.weights,
      recipe_limits(),
    )
    assert_true(wide_admitted.digest() != admitted.digest())

    let foreign_identity = runtime_recipe_fixture(
      plan=recipe_plan(
        content="4444444444444444444444444444444444444444444444444444444444444444",
      ),
    )
    expect_recipe_rule(
      () => {
        admit(
          foreign_identity.startup,
          inspection,
          repeated_inputs.launches,
          repeated_inputs.weights,
          recipe_limits(),
        )
      },
      WeightIdentity,
    )
    let foreign_schema = runtime_recipe_fixture(
      plan=recipe_plan(fp8_operations=[try! @plan.OperationId::new(2)]),
    )
    expect_recipe_rule(
      () => {
        admit(
          foreign_schema.startup,
          inspection,
          repeated_inputs.launches,
          repeated_inputs.weights,
          recipe_limits(),
        )
      },
      WeightSchema,
    )
    expect_recipe_rule(
      () => {
        admit(
          fixture.startup,
          changed_inspection,
          repeated_inputs.launches,
          repeated_inputs.weights,
          recipe_limits(),
        )
      },
      WeightStorage,
    )
    let wrong_order = runtime_inputs(
      fixture,
      inspection,
      claim_operation_delta=1,
    )
    expect_recipe_rule(
      () => {
        admit(
          fixture.startup,
          inspection,
          wrong_order.launches,
          wrong_order.weights,
          recipe_limits(),
        )
      },
      LaunchOrder,
    )
    let hostile_inputs : Array[(RuntimeRecipeInputs, Fp8RuntimeRecipeRule)] = [
      (
        runtime_inputs(fixture, inspection, weight_offset_delta=16L),
        WeightRegion,
      ),
      (
        runtime_inputs(fixture, inspection, weight_length_delta=-1L),
        WeightRegion,
      ),
      (runtime_inputs(fixture, inspection, weight_alignment=8L), WeightRegion),
      (
        runtime_inputs(fixture, inspection, scale_offset_delta=-16L),
        WeightRegion,
      ),
      (runtime_inputs(fixture, inspection, scale_alignment=8L), WeightRegion),
      (
        runtime_inputs(
          fixture,
          inspection,
          workspace=inspection.artifact_digest().as_hex(),
          workspace_bytes=inspection.layout().arena_bytes(),
        ),
        WorkspaceStorage,
      ),
      (
        runtime_inputs(fixture, inspection, workspace_bytes=2048L),
        WorkspaceCapacity,
      ),
    ]
    for hostile in hostile_inputs {
      expect_recipe_rule(
        () => {
          admit(
            fixture.startup,
            inspection,
            hostile.0.launches,
            hostile.0.weights,
            recipe_limits(),
          )
        },
        hostile.1,
      )
    }
    expect_recipe_rule(
      () => {
        admit(
          fixture.startup,
          inspection,
          repeated_inputs.launches,
          repeated_inputs.weights,
          recipe_limits(canonical=1),
        )
      },
      CanonicalBytes,
    )
    expect_scale_length_rejected_before_recipe(fixture, inspection)
    root.close()
  }
}
