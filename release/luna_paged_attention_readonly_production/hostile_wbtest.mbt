///|
fn readonly_production_fixture_identity(
  canary? : Int = 65521,
) -> ReadOnlyQualifiedIdentity raise {
  {
    candidate: "1".repeat(64),
    source: "2".repeat(64),
    recipe: "3".repeat(64),
    fallback_source: "4".repeat(64),
    fallback_recipe: "5".repeat(64),
    compiled_binding: "6".repeat(64),
    module_digest: "7".repeat(64),
    compile_receipt: "8".repeat(64),
    physical_evidence: "9".repeat(64),
    physical_files_manifest: "a".repeat(64),
    physical_outer_seal: "b".repeat(64),
    approved_physical_policy: "c".repeat(64),
    toolchain: "d".repeat(64),
    driver: "e".repeat(64),
    device_uuid: "GPU-01234567-89ab-cdef-0123-456789abcdef",
    device_pci_bus_id: "00000000:01:00.0",
    device_name_hex: "525458",
    device_total_memory_bytes: "17175674880",
    operation: 2,
    activation: 7,
    input_row_width: 6144,
    profile_id: 17,
    profile_rows: 64,
    profile_tokens: 260,
    profile_page_entries: 1024,
    tokens_per_page: 16,
    page_stride_bytes: 262144L,
    physical_page_capacity: 1024L,
    dispatch_canary: canary,
    dispatch_canary_cells: 260,
    symbol: "lf_paged_attention_readonly_103",
    abi: "readonly-abi-v1",
    target: @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
  }
}

///|
fn readonly_production_fixture_module() -> ApprovedPagedAttentionReadOnlyModule raise {
  let identity = readonly_production_fixture_identity()
  {
    manifest: readonly_production_digest(@utf8.encode("fixture-manifest")),
    identity,
    module_bytes: @utf8.encode("fixture-cubin"),
    launch: readonly_exact_launch(identity),
  }
}

///|
fn readonly_live_device_fixture(
  uuid? : String = "GPU-01234567-89ab-cdef-0123-456789abcdef",
) -> ReadOnlyLiveDeviceIdentity raise {
  {
    ordinal: 0,
    name_hex: "525458",
    uuid,
    pci: "00000000:01:00.0",
    total_memory_bytes: 17175674880L,
    capability: @device.DeviceCapability::from_observed_facts(
      compute_major=12,
      compute_minor=0,
      total_memory_bytes=17175674880L,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
  }
}

///|
test "approval canonical binds exact policy device and activation tuple" {
  let identity = readonly_production_fixture_identity()
  let canonical = readonly_approval_canonical(identity)
  let text = @utf8.decode(canonical)
  assert_true(
    text.contains("approved_physical_policy_sha256=" + "c".repeat(64)),
  )
  assert_true(
    text.contains("device_uuid=GPU-01234567-89ab-cdef-0123-456789abcdef"),
  )
  assert_true(text.contains("device_pci_bus_id=00000000:01:00.0"))
  assert_true(text.contains("activation_ref=7\ninput_row_width=6144"))
  assert_true(text.contains("dispatch_canary_cell_count=260"))
}

///|
test "exact eager launch is profile-sized and allocation-free metadata" {
  let launch = readonly_exact_launch(readonly_production_fixture_identity())
  assert_eq(launch.grid_x(), 260)
  assert_eq(launch.grid_y(), 1)
  assert_eq(launch.block_x(), 1)
  assert_eq(launch.shared_memory_bytes(), 0)
}

///|
test "canonical reader rejects duplicate and noncanonical numeric values" {
  try
    readonly_document_value(
      @utf8.encode("key=one\nkey=two\n"),
      "key",
      PhysicalPolicy,
    )
  catch {
    InvalidReadOnlyProduction(PhysicalPolicy) => ()
    _ => fail("unexpected duplicate-key error")
  } noraise {
    _ => fail("duplicate key reached approval")
  }
  for value in ["+1", "01", " 1", "1 "] {
    try ignore(readonly_exact_int(value, Operation)) catch {
      InvalidReadOnlyProduction(Operation) => ()
      _ => fail("unexpected noncanonical integer error")
    } noraise {
      _ => fail("noncanonical integer reached approval")
    }
  }
}

///|
test "approval digest changes with canary identity" {
  let first = readonly_production_digest(
    readonly_approval_canonical(readonly_production_fixture_identity()),
  )
  let second = readonly_production_digest(
    readonly_approval_canonical(
      readonly_production_fixture_identity(canary=65519),
    ),
  )
  assert_true(first != second)
}

///|
test "runtime-device record binds the opened-context identity projection" {
  let record = prepare_paged_attention_readonly_runtime_device_record_from_live_identity(
    readonly_production_fixture_module(),
    readonly_live_device_fixture(),
  )
  let canonical = @utf8.decode(record.canonical)
  assert_true(
    canonical.contains(
      "schema=lunaflux-paged-attention-readonly-runtime-device-record.v2",
    ),
  )
  assert_true(canonical.contains("live_device_ordinal=0"))
  assert_true(
    canonical.contains(
      "live_device_uuid=GPU-01234567-89ab-cdef-0123-456789abcdef",
    ),
  )
  assert_true(canonical.contains("observation=opened-context-device-identity"))
}

///|
test "runtime-device record rejects substituted live device identity" {
  try
    ignore(
      prepare_paged_attention_readonly_runtime_device_record_from_live_identity(
        readonly_production_fixture_module(),
        readonly_live_device_fixture(
          uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210",
        ),
      ),
    )
  catch {
    MismatchedReadOnlyProduction(RuntimeDevice) => ()
    _ => fail("unexpected substituted-device rejection")
  } noraise {
    _ => fail("substituted live device reached runtime-device signing record")
  }
}
