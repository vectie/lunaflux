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
  for
    altered in [
      text.replace(
        old="allow_approximate_prefill_exponential=0",
        new="allow_approximate_prefill_exponential=2",
      ),
      text.replace(old="row_variant_count=1", new="row_variant_count=2"),
      text.replace(old="row_variant_grid_x=4", new="row_variant_grid_x=3"),
      text.replace(
        old="row_variant_module_hex=736d616c6c",
        new="row_variant_module_hex=736d616c6d",
      ),
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
