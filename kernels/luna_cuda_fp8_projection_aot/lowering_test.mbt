///|
fn fp8_aot_plan() -> @plan.ModelPlan {
  let metadata = try! @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=1,
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
  try! @llama.build_paged_finite_e4m3_fp8_w8a8_v1(
    metadata,
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
}

///|
fn fp8_aot_profile(
  rows? : Int = 2,
  tokens? : Int = 8,
) -> @launch_contract.PagedKernelProfile {
  try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=rows,
    max_query_tokens=tokens,
    max_page_table_entries=16,
  )
}

///|
fn fp8_aot_target(major? : Int = 8, minor? : Int = 9) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn fp8_aot_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
      major=12,
      minor=8,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn fp8_aot_lower(
  operation : Int,
  profile? : @launch_contract.PagedKernelProfile = fp8_aot_profile(),
) -> Fp8ProjectionCudaLowering {
  let plan = fp8_aot_plan()
  let id = try! @plan.OperationId::new(operation)
  let operands = try! derive_fp8_projection_operands(
    plan,
    operation=id,
    profile,
  )
  try! lower_fp8_projection_cuda_aot_candidate(
    plan,
    operation=id,
    profile,
    fp8_aot_target(),
    fp8_aot_compiler(),
    operands,
    try! Fp8ProjectionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
}

///|
fn count_text(haystack : String, needle : String) -> Int {
  let mut count = 0
  let mut offset = 0
  while offset + needle.length() <= haystack.length() {
    if haystack[offset:offset + needle.length()].to_owned() == needle {
      count += 1
      offset += needle.length()
    } else {
      offset += 1
    }
  }
  count
}

///|
test "FP8 QKV AOT source is deterministic and owns exact scale ABI" {
  let first = fp8_aot_lower(2)
  let second = fp8_aot_lower(2)
  assert_eq(first, second)
  inspect(
    first.source_digest().as_hex(),
    content="b88549dbdfd78a1bd601a4b3fb392c6bdcfb400df977b79405d0d529f11a53a3",
  )
  inspect(
    first.recipe_digest().as_hex(),
    content="21ec1b296e3c45773a3b46526aab4d455ca11999a447db70e5d428416388a65a",
  )
  inspect(
    @crypto.bytes_to_hex_string(first.recipe_bytes()),
    content="736368656d613d6c756e61666c75782d6c756e612d637564612d6670382d70726f6a656374696f6e2d616f742d7265636970652d76310a66616d696c793d716b765f70726f6a656374696f6e0a6f7065726174696f6e5f69643d320a6f7065726174696f6e5f657865637574696f6e5f7368613235363d663233616661343461366261343162626333653065643237616461326365373463313432653538383633653434343863393734353834333730343162353735340a70726f66696c655f69643d310a656e7472795f706f696e745f69643d310a66756e6374696f6e5f73796d626f6c3d6c756e61666c75785f6c756e615f716b765f70726f6a656374696f6e5f6670385f65346d335f773861385f72656c656173655f76310a736f757263655f7368613235363d623838353439646264666437386131626436303161346233666233393263366264636662343030646639373762373934303564306435323966313161353361330a746f6f6c636861696e5f7368613235363d313233343536373839306162636465663132333435363738393061626364656631323334353637383930616263646566313233343536373839306162636465660a636f6d70696c65725f76657273696f6e3d31322e382e310a7461726765743d736d5f38390a6f75747075743d637562696e0a72657175697265735f637564615f6670385f6865616465723d747275650a72657175697265735f6e61746976655f65346d335f636f6e76657273696f6e3d747275650a65346d335f636f6e76657273696f6e3d5f5f6e765f6376745f666c6f61745f746f5f6670385f73617466696e6974655f726e655f76310a6f7074696d697a6174696f6e3d330a666d61643d66616c73650a72656173736f63696174653d66616c73650a6d61785f7265676973746572733d3132380a696e7075745f77696474683d380a6f75747075745f77696474683d31360a696e7465726d6564696174655f77696474683d340a6d61785f71756572795f726f77733d320a6d61785f71756572795f746f6b656e733d380a6d61785f706167655f7461626c655f656e74726965733d31360a677269643d312c312c310a626c6f636b3d3235362c312c310a6e756d657269633d626631365f696e7075745f64796e616d69635f7065725f74656e736f725f6633325f7363616c655f66696e6974655f65346d335f7765696768745f7065725f74656e736f725f6633325f7363616c655f6633325f616363756d756c61746f725f626631365f6f75747075745f76310a7261775f706172616d657465725f6162693d63616e6f6e6963616c5f70616765645f76345f6f706572616e64735f76310a61637469766174696f6e5f7363616c655f776f726b73706163655f62797465733d340a61637469766174696f6e5f7363616c655f776f726b73706163655f616c69676e6d656e743d340a776f726b73706163655f726f6c653d61637469766174696f6e5f7363616c655f776f726b73706163655f6633325f7363616c61720a6f706572616e64733d72756e74696d653a737465705f636f756e74734032302f342c61637469766174696f6e5f696e3a31403132382f31362c7765696768743a324036342f31362c7765696768745f7363616c653a323a313240342f31362c7765696768743a334033322f31362c7765696768745f7363616c653a333a313340342f31362c7765696768743a344033322f31362c7765696768745f7363616c653a343a313440342f31362c61637469766174696f6e5f6f75743a32403235362f31362c776f726b737061636540342f340a",
  )
  assert_eq(first.family(), QkvProjection)
  assert_eq(
    first.function_symbol(),
    "lunaflux_luna_qkv_projection_fp8_e4m3_w8a8_release_v1",
  )
  let source = try! @utf8.decode(first.source_bytes())
  assert_true(source.contains("#include <cuda_fp8.h>"))
  assert_true(source.contains("CUDA_VERSION < 12000"))
  assert_true(source.contains("__CUDA_ARCH__ != 890"))
  assert_true(source.contains("__CUDA_ARCH__ != 900"))
  assert_true(
    source.contains(
      "__nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3)",
    ),
  )
  assert_true(source.contains("fminf(448.0f, fmaxf(-448.0f, value))"))
  assert_true(source.contains("bounded == 0.0f ? 0.0f : bounded"))
  assert_eq(
    count_text(source, "*activation_scale_workspace = activation_scale"),
    1,
  )
  assert_true(source.contains("if (index == 0ULL)"))
  let operands = first.operands()
  assert_eq(operands.length(), 10)
  assert_true(operands[3].role() is WeightScaleInput(_, _))
  assert_true(operands[5].role() is WeightScaleInput(_, _))
  assert_true(operands[7].role() is WeightScaleInput(_, _))
  assert_eq(operands[9].role(), Workspace)
  assert_eq(operands[9].byte_count(), 4L)
  let recipe = try! @utf8.decode(first.recipe_bytes())
  assert_true(
    recipe.contains("raw_parameter_abi=canonical_paged_v4_operands_v1"),
  )
  assert_true(
    recipe.contains(
      "operation_execution_sha256=" + first.execution_digest().as_hex(),
    ),
  )
}

///|
test "FP8 dense and head identities cannot replay across operation or profile" {
  let dense = fp8_aot_lower(5)
  let head = fp8_aot_lower(11)
  assert_eq(dense.family(), DenseProjection)
  assert_eq(head.family(), LanguageModelHead)
  assert_true(dense.function_symbol() != head.function_symbol())
  assert_true(dense.source_digest() != head.source_digest())
  let batch_one = fp8_aot_lower(5, profile=fp8_aot_profile(rows=1, tokens=8))
  let batch_two = fp8_aot_lower(5, profile=fp8_aot_profile(rows=2, tokens=8))
  assert_eq(batch_one.source_digest(), batch_two.source_digest())
  assert_true(batch_one.recipe_digest() != batch_two.recipe_digest())
}

///|
test "FP8 gated MLP fails closed until a second dynamic-scale ABI exists" {
  let plan = fp8_aot_plan()
  let id = try! @plan.OperationId::new(8)
  try
    derive_fp8_projection_operands(plan, operation=id, fp8_aot_profile())
  catch {
    Unsupported(CompoundGatedActivationScale) => ()
    _ => fail("unexpected gated rejection")
  } noraise {
    _ => fail("mixed FP8/F32 gated MLP source was emitted")
  }
}

///|
test "FP8 candidate rejects target substitution and operand substitution" {
  let plan = fp8_aot_plan()
  let dense = try! @plan.OperationId::new(5)
  let qkv = try! @plan.OperationId::new(2)
  let profile = fp8_aot_profile()
  let dense_operands = try! derive_fp8_projection_operands(
    plan,
    operation=dense,
    profile,
  )
  try
    lower_fp8_projection_cuda_aot_candidate(
      plan,
      operation=dense,
      profile,
      fp8_aot_target(major=8, minor=0),
      fp8_aot_compiler(),
      dense_operands,
      try! Fp8ProjectionAotLimits::new(
        max_source_bytes=65536,
        max_recipe_bytes=65536,
      ),
    )
  catch {
    Unsupported(DeviceTarget) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("sm80 FP8 candidate was emitted")
  }
  let qkv_operands = try! derive_fp8_projection_operands(
    plan,
    operation=qkv,
    profile,
  )
  try
    lower_fp8_projection_cuda_aot_candidate(
      plan,
      operation=dense,
      profile,
      fp8_aot_target(),
      fp8_aot_compiler(),
      qkv_operands,
      try! Fp8ProjectionAotLimits::new(
        max_source_bytes=65536,
        max_recipe_bytes=65536,
      ),
    )
  catch {
    Mismatch(OperandOrder) => ()
    _ => fail("unexpected operand rejection")
  } noraise {
    _ => fail("QKV operands replayed as dense")
  }
}
