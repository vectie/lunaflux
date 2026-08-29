///|
fn fused_physical_test_replace(
  canonical : Bytes,
  old : String,
  replacement : String,
) -> Bytes {
  let document = @utf8.decode(canonical) catch {
    _ => abort("hostile physical-evidence fixture was not UTF-8")
  }
  if !document.contains(old) {
    abort("hostile physical-evidence fixture missed substitution target")
  }
  @utf8.encode(document.replace(old~, new=replacement))
}

///|
fn fused_physical_test_reject(
  fixture : FusedPhysicalBlackBoxFixture,
  canonical : Bytes,
  label : String,
) -> Unit raise {
  try fused_physical_test_admit(fixture, canonical) catch {
    _ => ()
  } noraise {
    _ => fail("hostile physical evidence was admitted: " + label)
  }
}

///|
test "physical fused evidence rejects duplicate unknown reordered missing and overflow fields" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  let schema = "schema=lunaflux-fused-physical-qualification-evidence.v1\n"
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(canonical, schema, schema + schema),
    "duplicate",
  )
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(
      canonical, "family_count=2", "unknown_family_count=2",
    ),
    "unknown",
  )
  let reordered = @utf8.decode(canonical)
    .replace(old="family_count=2\n", new="temporary-order-marker\n")
    .replace(old="referee_shape_count=4\n", new="family_count=2\n")
    .replace(old="temporary-order-marker\n", new="referee_shape_count=4\n")
  fused_physical_test_reject(fixture, @utf8.encode(reordered), "reordered")
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(
      canonical, "residual_family=residual-rmsnorm\n", "",
    ),
    "missing-family",
  )
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(canonical, "family_count=2", "family_count=02"),
    "noncanonical-number",
  )
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(
      canonical, "stderr_bytes=0", "stderr_bytes=999999999999999999999999999999999999",
    ),
    "overflow",
  )
  fused_physical_test_reject(
    fixture,
    @utf8.encode("x".repeat(131073)),
    "canonical-limit",
  )
}

///|
test "physical fused evidence rejects coherently repinned identity substitutions" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  let qkv = fixture.qkv
  let substitutions = [
    (
      "qkv_family=qkv-positioned-rope-paged-kvwrite", "qkv_family=residual-rmsnorm",
    ),
    (
      "qkv_candidate_sha256=" + qkv.candidate_digest().as_hex(),
      "qkv_candidate_sha256=" + "1".repeat(64),
    ),
    (
      "qkv_source_sha256=" + qkv.source_digest().as_hex(),
      "qkv_source_sha256=" + "2".repeat(64),
    ),
    (
      "qkv_recipe_sha256=" + qkv.recipe_digest().as_hex(),
      "qkv_recipe_sha256=" + "3".repeat(64),
    ),
    (
      "qkv_toolchain_sha256=" +
      qkv.compiler_policy().identity().toolchain_digest_sha256(),
      "qkv_toolchain_sha256=" + "4".repeat(64),
    ),
    ("qkv_compiler_major=13", "qkv_compiler_major=14"),
    ("qkv_target=sm_120", "qkv_target=sm_121"),
    ("qkv_grid_x=8", "qkv_grid_x=9"),
    (
      "qkv_function_symbol=" + qkv.function_symbol(),
      "qkv_function_symbol=lunaflux_substituted_fused_symbol",
    ),
    (
      "qkv_typed_abi=" +
      @luna_cuda_fused_parallel_aot.QkvPositionedRopePagedKvWriteAbiV1.canonical_name(),
      "qkv_typed_abi=step_counts,substituted",
    ),
    (
      "qkv_first_build_cubin_sha256=" +
      fixture.qkv_compiled.module_digest().as_hex(),
      "qkv_first_build_cubin_sha256=" + "5".repeat(64),
    ),
    (
      "qkv_compile_receipt_sha256=" +
      fixture.qkv_compiled.compile_receipt_digest().as_hex(),
      "qkv_compile_receipt_sha256=" + "6".repeat(64),
    ),
    ("qkv_deterministic=true", "qkv_deterministic=false"),
  ]
  for index, substitution in substitutions {
    let (old, replacement) = substitution
    fused_physical_test_reject(
      fixture,
      fused_physical_test_replace(canonical, old, replacement),
      "identity-" + index.to_string(),
    )
  }
  let cubin_digest = fixture.qkv_compiled.module_digest().as_hex()
  let coherent_cubin = @utf8.decode(canonical)
    .replace(
      old="qkv_first_build_cubin_sha256=" + cubin_digest,
      new="qkv_first_build_cubin_sha256=" + "7".repeat(64),
    )
    .replace(
      old="qkv_second_build_cubin_sha256=" + cubin_digest,
      new="qkv_second_build_cubin_sha256=" + "7".repeat(64),
    )
  fused_physical_test_reject(
    fixture,
    @utf8.encode(coherent_cubin),
    "coherent-cubin",
  )
}

///|
test "physical fused evidence rejects widened numerics canary sanitizer and race claims" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  let widened = @utf8.decode(canonical)
    .replace(
      old="qkv_absolute_tolerance_ppb=250000",
      new="qkv_absolute_tolerance_ppb=250001",
    )
    .replace(
      old="qkv_maximum_absolute_error_ppb=250000",
      new="qkv_maximum_absolute_error_ppb=250001",
    )
  fused_physical_test_reject(
    fixture,
    @utf8.encode(widened),
    "widened-tolerance",
  )
  for
    index, substitution in [
      (
        "qkv_observed_maximum_absolute_error_ppb=1000", "qkv_observed_maximum_absolute_error_ppb=250001",
      ),
      (
        "qkv_observed_maximum_relative_error_ppb=2000", "qkv_observed_maximum_relative_error_ppb=500001",
      ),
      ("qkv_dispatch_canary_pass=true", "qkv_dispatch_canary_pass=false"),
      ("qkv_sanitizer_status=pass", "qkv_sanitizer_status=fail"),
      ("qkv_race_status=pass", "qkv_race_status=fail"),
    ] {
    let (old, replacement) = substitution
    fused_physical_test_reject(
      fixture,
      fused_physical_test_replace(canonical, old, replacement),
      "qualification-" + index.to_string(),
    )
  }
  let canary = (fixture.qkv.dispatch_canary() * 12U).to_string()
  fused_physical_test_reject(
    fixture,
    fused_physical_test_replace(
      canonical,
      "qkv_observed_dispatch_canary=" + canary,
      "qkv_observed_dispatch_canary=1",
    ),
    "canary-value",
  )
}

///|
test "physical fused evidence rejects nonzero resources stderr and unsealed evidence" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  for
    index, substitution in [
      ("stderr_bytes=0", "stderr_bytes=1"),
      ("contexts_live_after=0", "contexts_live_after=1"),
      ("streams_live_after=0", "streams_live_after=1"),
      ("events_live_after=0", "events_live_after=1"),
      ("graphs_live_after=0", "graphs_live_after=1"),
      ("graph_execs_live_after=0", "graph_execs_live_after=1"),
      ("allocations_live_after=0", "allocations_live_after=1"),
      ("modules_live_after=0", "modules_live_after=1"),
      ("device_bytes_live_after=0", "device_bytes_live_after=1"),
      ("pending_work_after=0", "pending_work_after=1"),
      ("artifact_directory_sealed=true", "artifact_directory_sealed=false"),
      ("evidence_directory_sealed=true", "evidence_directory_sealed=false"),
      (
        "artifact_directory_seal_sha256=" + "e".repeat(64),
        "artifact_directory_seal_sha256=" + "0".repeat(64),
      ),
      ("manifest_bindable=false", "manifest_bindable=true"),
      ("promotion_authority=absent", "promotion_authority=present"),
    ] {
    let (old, replacement) = substitution
    fused_physical_test_reject(
      fixture,
      fused_physical_test_replace(canonical, old, replacement),
      "closure-" + index.to_string(),
    )
  }
}

///|
test "physical fused evidence rejects independently pinned digest mismatch" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  let changed = fused_physical_test_replace(
    canonical, "pending_work_after=0", "pending_work_after=1",
  )
  try
    admit_fused_physical_evidence(
      fixture.qkv,
      fixture.qkv_compiled,
      fixture.residual,
      fixture.residual_compiled,
      changed,
      expected_digest=fused_physical_test_digest(canonical),
      expected_approved_policy_digest=fused_physical_test_policy_digest(),
    )
  catch {
    @luna_fused_physical_evidence.FusedPhysicalEvidenceMismatch(Digest) => ()
    _ => fail("unexpected outer digest rejection")
  } noraise {
    _ => fail("changed evidence matched the independent digest")
  }
}
