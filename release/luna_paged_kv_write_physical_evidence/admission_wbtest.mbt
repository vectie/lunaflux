///|
fn kv_fixture_admit(
  fixture : KvPhysicalFixture,
  canonical : Bytes,
) -> QualifiedPagedKvWriteArtifact raise {
  admit_qualified_paged_kv_write_artifact(
    fixture.candidate,
    fixture.compiled,
    canonical,
    expected_evidence_digest=kv_physical_sha256(canonical),
    expected_approved_policy_digest=kv_fixture_digest("9".repeat(64)),
  )
}

///|
fn kv_fixture_replace(
  canonical : Bytes,
  old : String,
  replacement : String,
) -> Bytes {
  let text = @utf8.decode(canonical) catch {
    _ => abort("fixture canonical is not UTF-8")
  }
  if !text.contains(old) {
    abort("hostile substitution target is absent")
  }
  @utf8.encode(text.replace(old~, new=replacement))
}

///|
test "qualified paged KV-write artifact retains exact inert identities" {
  let fixture = kv_physical_fixture()
  let canonical = kv_physical_fixture_canonical(fixture)
  let qualified = kv_fixture_admit(fixture, canonical)
  assert_eq(qualified.candidate_digest(), fixture.candidate.candidate_digest())
  assert_eq(qualified.module_digest(), fixture.compiled.module_digest())
  assert_eq(
    qualified.compiled_binding_digest(),
    fixture.compiled.binding_digest(),
  )
  assert_eq(qualified.evidence_digest(), kv_physical_sha256(canonical))
  assert_eq(
    qualified.approved_policy_digest(),
    kv_fixture_digest("9".repeat(64)),
  )
  assert_eq(qualified.positioned_qkv_row_width(), 256)
  assert_eq(qualified.device_uuid(), "GPU-test-sm120")
  assert_eq(qualified.device_pci(), "00000000:01:00.0")
  assert_eq(qualified.dispatch_canary_per_token(), 22321U)
  assert_false(qualified.manifest_bindable())
  assert_false(qualified.promotion_authority())
  assert_false(qualified.runtime_dispatch_authority())
}

///|
test "paged KV-write evidence rejects policy oracle canary resource and authority drift" {
  let fixture = kv_physical_fixture()
  let canonical = kv_physical_fixture_canonical(fixture)
  for
    pair in [
      (
        "approved_policy_sha256=" + "9".repeat(64),
        "approved_policy_sha256=" + "8".repeat(64),
      ),
      (
        "candidate_sha256=" + fixture.candidate.candidate_digest().as_hex(),
        "candidate_sha256=" + "7".repeat(64),
      ),
      ("mixed_case_pass=pass", "mixed_case_pass=fail"),
      ("observed_key_mutations=512", "observed_key_mutations=0"),
      ("observed_dispatch_canary=178568", "observed_dispatch_canary=0"),
      ("guards_unchanged=pass", "guards_unchanged=fail"),
      (
        "nvdisasm_sass_instruction_count=17", "nvdisasm_sass_instruction_count=16",
      ),
      ("sass_global_load_observed=true", "sass_global_load_observed=false"),
      ("local_bytes=0", "local_bytes=1"),
      ("initcheck_status=pass", "initcheck_status=fail"),
      (
        "context_device_uuid=GPU-test-sm120", "context_device_uuid=GPU-replayed-sm120",
      ),
      ("physical_cuda_observed=true", "physical_cuda_observed=false"),
      ("synthetic_test_only=false", "synthetic_test_only=true"),
      ("manifest_bindable=false", "manifest_bindable=true"),
    ] {
    try
      kv_fixture_admit(fixture, kv_fixture_replace(canonical, pair.0, pair.1))
    catch {
      _ => ()
    } noraise {
      _ => fail("hostile physical evidence was admitted: " + pair.0)
    }
  }
}

///|
test "paged KV-write evidence rejects partial replay and independently pinned digest drift" {
  let fixture = kv_physical_fixture()
  let canonical = kv_physical_fixture_canonical(fixture)
  let text = @utf8.decode(canonical) catch {
    _ => abort("fixture canonical is not UTF-8")
  }
  let missing = @utf8.encode(
    text.replace(old="page_tail_case_pass=pass\n", new=""),
  )
  try kv_fixture_admit(fixture, missing) catch {
    _ => ()
  } noraise {
    _ => fail("partial physical evidence was admitted")
  }
  try
    admit_qualified_paged_kv_write_artifact(
      fixture.candidate,
      fixture.compiled,
      canonical,
      expected_evidence_digest=kv_fixture_digest("8".repeat(64)),
      expected_approved_policy_digest=kv_fixture_digest("9".repeat(64)),
    )
  catch {
    Mismatch(Digest) => ()
    _ => fail("unexpected independent digest rejection")
  } noraise {
    _ => fail("wrong independently expected digest was admitted")
  }
}
