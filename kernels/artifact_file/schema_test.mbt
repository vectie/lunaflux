///|
test "reader rejects unknown duplicate and malformed fields before admission" {
  let contracts = fixture_contracts()
  let limits = fixture_file_limits()
  let unknown = manifest_json(
    [
      module_json(fixture_digest().sha256(), "kernel.cubin", [
        entry_json(1, 11, "embedding_prefill_2"),
        entry_json(1, 12, "embedding_prefill_4"),
      ]),
    ],
    extra_root=",\"execute\":\"now\"",
  )
  try @artifact_file.read_manifest(unknown, contracts, limits) catch {
    UnknownField(area=ManifestRoot, index=5) => ()
    _ => fail("expected unknown root field")
  } noraise {
    _ => fail("expected failure")
  }

  let duplicate = @utf8.encode(
    "{\"schema_version\":1,\"schema_version\":1,\"model_identity\":{},\"device_target\":{},\"catalog_version\":1,\"modules\":[]}",
  )
  try @artifact_file.read_manifest(duplicate, contracts, limits) catch {
    DuplicateField(depth=1, index=1) => ()
    _ => fail("expected duplicate field")
  } noraise {
    _ => fail("expected failure")
  }

  let unknown_module = @utf8.encode(
    (try! @utf8.decode(valid_manifest())).replace(
      old="\"entry_points\":[",
      new="\"cache_path\":\"x\",\"entry_points\":[",
    ),
  )
  try @artifact_file.read_manifest(unknown_module, contracts, limits) catch {
    UnknownField(area=ManifestModule, index=2) => ()
    _ => fail("expected unknown module field")
  } noraise {
    _ => fail("expected failure")
  }
}

///|
test "reader pins schema model device catalog and entry identities" {
  let contracts = fixture_contracts()
  let limits = fixture_file_limits()
  let cases = [
    ("\"schema_version\":1", "\"schema_version\":2", "schema"),
    ("\"compute_major\":8", "\"compute_major\":9", "device"),
    ("\"catalog_version\":1", "\"catalog_version\":2", "catalog"),
  ]
  for test_case in cases {
    let mutated = @utf8.encode(
      (try! @utf8.decode(valid_manifest())).replace(
        old=test_case.0,
        new=test_case.1,
      ),
    )
    try @artifact_file.read_manifest(mutated, contracts, limits) catch {
      InvalidField(
        area=ManifestRoot,
        field=SchemaVersionField,
        issue=UnsupportedValue,
        index=0
      ) if test_case.2 == "schema" => ()
      IdentityMismatch(DeviceComputeMajor) if test_case.2 == "device" => ()
      IdentityMismatch(KernelCatalogVersion) if test_case.2 == "catalog" => ()
      _ => fail("unexpected identity rejection")
    } noraise {
      _ => fail("expected failure")
    }
  }

  let content_mutation = @utf8.encode(
    (try! @utf8.decode(valid_manifest())).replace(
      old=fixture_identity().content().as_hex(),
      new="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    ),
  )
  try @artifact_file.read_manifest(content_mutation, contracts, limits) catch {
    IdentityMismatch(ModelContentIdentity) => ()
    _ => fail("expected model identity rejection")
  } noraise {
    _ => fail("expected failure")
  }

  let unknown_entry = valid_manifest(entry_ids=[11, 99])
  let parsed = @artifact_file.read_manifest(unknown_entry, contracts, limits)
  assert_eq(parsed.entry_point_count(), 2)
  try
    @artifact.admit(
      contracts,
      [
        @artifact.KernelModuleInput::from_bytes(
          digest=fixture_digest(),
          module_bytes=fixture_module_bytes(),
        ),
      ],
      [
        @artifact.KernelEntryPointInput::new(
          entry_point=fixture_entry(fixture_digest(), 11),
          function_symbol="embedding_prefill_2",
        ),
        @artifact.KernelEntryPointInput::new(
          entry_point=fixture_entry(fixture_digest(), 99),
          function_symbol="embedding_prefill_4",
        ),
      ],
      limits.artifact(),
    )
  catch {
    UnreferencedEntryPoint(identity) =>
      assert_eq(identity.entry_point_id().as_int(), 99)
    _ => fail("expected semantic entry rejection")
  } noraise {
    _ => fail("expected failure")
  }
}

///|
test "manifest byte path count and JSON depth bounds precede allocation" {
  let contracts = fixture_contracts()
  let manifest = valid_manifest()
  try
    @artifact_file.read_manifest(
      manifest,
      contracts,
      fixture_file_limits(max_manifest_bytes=16L),
    )
  catch {
    FileTooLarge(file=ManifestFile, maximum=16L) => ()
    _ => fail("expected manifest byte limit")
  } noraise {
    _ => fail("expected failure")
  }
  try
    @artifact_file.read_manifest(
      valid_manifest(path="too-long-kernel.cubin"),
      contracts,
      fixture_file_limits(max_path_bytes=8),
    )
  catch {
    InvalidPath(file=ModuleFile(0), issue=PathTooLong) => ()
    _ => fail("expected path byte limit")
  } noraise {
    _ => fail("expected failure")
  }
  let duplicate_modules = manifest_json([
    module_json(fixture_digest().sha256(), "one.cubin", [
      entry_json(1, 11, "embedding_prefill_2"),
    ]),
    module_json(fixture_digest().sha256(), "two.cubin", [
      entry_json(1, 12, "embedding_prefill_4"),
    ]),
  ])
  try
    @artifact_file.read_manifest(
      duplicate_modules,
      contracts,
      fixture_file_limits(max_modules=1),
    )
  catch {
    InvalidField(
      area=ManifestRoot,
      field=ModulesField,
      issue=CountExceeded,
      index=0
    ) => ()
    _ => fail("expected module count limit")
  } noraise {
    _ => fail("expected failure")
  }
}

///|
test "symbol byte bound is enforced during manifest preflight" {
  let manifest = manifest_json([
    module_json(fixture_digest().sha256(), "kernel.cubin", [
      entry_json(1, 11, "symbol_that_is_too_long"),
      entry_json(1, 12, "ok"),
    ]),
  ])
  let bounded_artifact = try! @artifact.KernelArtifactLimits::new(
    max_modules=4,
    max_entry_points=8,
    max_module_bytes=1024L,
    max_total_module_bytes=2048L,
    max_function_symbol_bytes=8,
  )
  let bounded_file = try! @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=16384L,
    max_json_depth=8,
    max_path_bytes=128,
    artifact=bounded_artifact,
  )
  try
    @artifact_file.read_manifest(manifest, fixture_contracts(), bounded_file)
  catch {
    InvalidField(
      area=ManifestEntryPoint,
      field=FunctionSymbolField,
      issue=StringTooLong,
      index=0
    ) => ()
    _ => fail("expected bounded symbol rejection")
  } noraise {
    _ => fail("expected failure")
  }
}
