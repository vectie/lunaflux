///|
async test "paged manifest loads through the exact filesystem admission path" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-paged-aot-file-valid")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let manifest = paged_valid_manifest()
    write_file(root, "manifest.json", manifest)
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    let contracts = paged_file_contracts()
    let parsed = @artifact_file.read_paged_manifest(
      manifest,
      contracts,
      fixture_file_limits(),
    )
    assert_eq(parsed.module_count(), 1)
    assert_eq(parsed.entry_point_count(), 2)
    let bundle = @artifact_file.load_paged_kv(
      authority,
      "manifest.json",
      expected_manifest(manifest),
      contracts,
      fixture_file_limits(),
    )
    assert_eq(bundle.model_identity(), contracts.model_identity())
    assert_eq(bundle.device_target(), contracts.device_target())
    assert_eq(bundle.catalog_version(), @catalog.CatalogVersion::v3())
    assert_eq(bundle.module_count(), 1)
    assert_eq(bundle.entry_point_count(), 2)
    assert_eq(bundle.modules()[0].module_bytes(), fixture_module_bytes())
    authority.close()
  }
}

///|
async test "direct paged source reuses the approved module loader" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-paged-aot-direct")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let digest = fixture_digest()
    let source = @artifact_file.KernelArtifactManifest::admit_files(
      [
        @artifact_file.KernelModuleFileInput::new(
          digest~,
          path_identifier="kernel.cubin",
        ),
      ],
      [
        @artifact.KernelEntryPointInput::new(
          entry_point=paged_file_entry(digest, 1),
          function_symbol="paged_rotary",
        ),
        @artifact.KernelEntryPointInput::new(
          entry_point=paged_file_entry(digest, 2),
          function_symbol="paged_attention",
        ),
      ],
      fixture_file_limits(),
    )
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    let bundle = @artifact_file.load_paged_kv_admitted(
      authority,
      source,
      paged_file_contracts(),
      fixture_file_limits(),
    )
    assert_eq(bundle.module_count(), 1)
    assert_eq(bundle.entry_point_count(), 2)
    assert_false(authority.is_closed())
    authority.close()
  }
}

///|
async test "paged filesystem admission fails closed on version and exports" {
  let contracts = paged_file_contracts()
  let wrong_version = manifest_json(
    [
      module_json(fixture_digest().sha256(), "kernel.cubin", [
        entry_json(11, 21, "paged_rotary"),
        entry_json(12, 22, "paged_attention"),
      ]),
    ],
    catalog_version=2,
    plan=paged_file_plan(),
  )
  try
    @artifact_file.read_paged_manifest(
      wrong_version,
      contracts,
      fixture_file_limits(),
    )
  catch {
    IdentityMismatch(KernelCatalogVersion) => ()
    _ => fail("unexpected paged catalog-version error")
  } noraise {
    _ => fail("expected paged catalog-version mismatch")
  }

  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-paged-aot-file-missing")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let missing_export = paged_valid_manifest(entry_count=1)
    write_file(root, "manifest.json", missing_export)
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    try
      @artifact_file.load_paged_kv(
        authority,
        "manifest.json",
        expected_manifest(missing_export),
        contracts,
        fixture_file_limits(),
      )
    catch {
      ArtifactRejected(MissingEntryPoint(found)) =>
        assert_eq(found, paged_file_entry(fixture_digest(), 2))
      _ => fail("unexpected paged missing-entry error")
    } noraise {
      _ => fail("expected missing paged entry point")
    }
    authority.close()
  }
}
