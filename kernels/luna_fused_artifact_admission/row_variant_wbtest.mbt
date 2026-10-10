///|
fn row_variant_base(symbol : String) -> ReusableFusedRuntimeModuleInput raise {
  ReusableFusedRuntimeModuleInput::new(
    abi=AttentionIngressPreparedRotaryProductionV11,
    first_operation=2,
    operation_count=4,
    function_symbol=symbol,
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=32,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    module_bytes=b"base",
    query_tile_rows=64,
  )
}

///|
fn row_variant_fixture(
  maximum : Int,
  rows : Int,
) -> ReusableFusedRuntimeRowVariant raise {
  ReusableFusedRuntimeRowVariant::new(
    maximum_token_rows=maximum,
    query_tile_rows=rows,
    function_symbol="ingress",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=4,
      grid_y=32,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    module_bytes=b"small",
  )
}

///|
test "row alternatives roundtrip without changing legacy serialization" {
  let base = row_variant_base("ingress")
  let variant = row_variant_fixture(64, 16)
  let input = { ..base, row_variant: Some(variant), }
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[input],
  )
  assert_true(
    @utf8.decode(bytes).has_prefix(
      "schema=lunaflux-reusable-fused-runtime-bundle.v13\n",
    ),
  )
  let admitted = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_true(admitted.modules()[0].row_variant() == Some(variant))
  let text = @utf8.decode(bytes)
  assert_true(text.contains("allow_approximate_prefill_exponential=0\n"))
  let changed_payload = @utf8.encode(
    text.replace(
      old="row_variant_module_hex=736d616c6c",
      new="row_variant_module_hex=736d616c6d",
    ),
  )
  let labelled = admit_digest_pinned_reusable_fused_runtime_bundle(
    changed_payload,
    "f".repeat(64),
  )
  guard labelled.modules()[0].row_variant() is Some(retained) else {
    fail("row variant was lost")
  }
  assert_eq(retained.module_bytes, b"smalm")
  assert_eq(retained.module_label, variant.module_label)
  for
    altered in [
      text.replace(
        old="allow_approximate_prefill_exponential=0",
        new="allow_approximate_prefill_exponential=2",
      ),
      text.replace(old="row_variant_count=1", new="row_variant_count=2"),
      text.replace(old="row_variant_grid_x=4", new="row_variant_grid_x=3"),
      text.replace(old="bundle.v13", new="bundle.v12"),
    ] {
    let bad = @utf8.encode(altered)
    try
      admit_digest_pinned_reusable_fused_runtime_bundle(
        bad,
        @crypto.bytes_to_hex_string(@crypto.sha256(bad)),
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("malformed row variant accepted")
    }
  }
  try
    reusable_runtime_validate_row_variant(
      ResidualRmsNormProductionV2,
      "ingress",
      base.launch,
      1,
      Some(variant),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("residual ABI acquired ingress operands")
  }
}

///|
fn row_scope_reader(
  abi : ReusableFusedRuntimeKernelAbi,
  symbol : String,
) -> ReusableFusedRuntimeModuleInput raise {
  ReusableFusedRuntimeModuleInput::new(
    abi~,
    first_operation=5,
    operation_count=1,
    function_symbol=symbol,
    module_bytes=b"scope-reader",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=1,
      grid_z=1,
      block_x=32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
  )
}

///|
test "attention route scope consumes variant label without payload replay" {
  let variant = row_variant_fixture(64, 16)
  let ingress = { ..row_variant_base("ingress"), row_variant: Some(variant), }
  let modules = [
    ingress,
    row_scope_reader(QwenPagedAttentionReadOnlyProductionV2, "base"),
    row_scope_reader(QwenPagedAttentionPrefillWideQueryProductionV5, "wide"),
  ]
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules~,
  )
  let label = "f".repeat(64)
  let original = admit_digest_pinned_reusable_fused_runtime_bundle(bytes, label)
  let changed = admit_digest_pinned_reusable_fused_runtime_bundle(
    @utf8.encode(
      @utf8.decode(bytes).replace(
        old="row_variant_module_hex=736d616c6c",
        new="row_variant_module_hex=736d616c6d",
      ),
    ),
    label,
  )
  assert_eq(original.attention_route_scope(), changed.attention_route_scope())
  let different_label = admit_digest_pinned_reusable_fused_runtime_bundle(
    @utf8.encode(
      @utf8.decode(bytes).replace(
        old="row_variant_module_sha256=" + variant.module_label.as_hex(),
        new="row_variant_module_sha256=" + "e".repeat(64),
      ),
    ),
    label,
  )
  assert_true(
    original.attention_route_scope() != different_label.attention_route_scope(),
  )
  let different_geometry = admit_digest_pinned_reusable_fused_runtime_bundle(
    @utf8.encode(
      @utf8.decode(bytes)
      .replace(
        old="row_variant_query_tile_rows=16",
        new="row_variant_query_tile_rows=32",
      )
      .replace(old="row_variant_grid_x=4", new="row_variant_grid_x=2"),
    ),
    label,
  )
  assert_true(
    original.attention_route_scope() !=
    different_geometry.attention_route_scope(),
  )
}

///|
test "row alternatives reject invalid coverage and same-ABI launch mismatch" {
  for (maximum, rows) in [(1, 16), (64, 0), (64, 128), (63, 32)] {
    try row_variant_fixture(maximum, rows) catch {
      _ => ()
    } noraise {
      _ => fail("invalid extent accepted")
    }
  }
  let v = row_variant_fixture(64, 16)
  let base = row_variant_base("wrong")
  try
    reusable_runtime_validate_row_variant(
      base.abi,
      base.symbol,
      base.launch,
      1,
      Some(v),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("different symbol accepted")
  }
}
