///|
async test "valid exact manifest admits one multi-function module" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-file-valid")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let manifest = valid_manifest()
    write_file(root, "manifest.json", manifest)
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    let bundle = @artifact_file.load(
      authority,
      "manifest.json",
      expected_manifest(manifest),
      fixture_contracts(),
      fixture_file_limits(),
    )
    assert_eq(bundle.module_count(), 1)
    assert_eq(bundle.entry_point_count(), 2)
    assert_eq(bundle.modules()[0].module_bytes(), fixture_module_bytes())
    assert_eq(bundle.entry_points()[0].function_symbol(), "embedding_prefill_2")
    assert_eq(bundle.entry_points()[1].function_symbol(), "embedding_prefill_4")
    assert_false(authority.is_closed())
    authority.close()
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(manifest),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      FileIo(file=ManifestFile, operation=OpenFile, issue=AuthorityClosed) => ()
      _ => fail("unexpected closed-root mapping")
    } noraise {
      _ => fail("closed approved root was reused")
    }
  }
}

///|
async test "direct stateless source loads without a JSON filesystem manifest" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-direct-source")
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
          entry_point=fixture_entry(digest, 11),
          function_symbol="embedding_prefill_2",
        ),
        @artifact.KernelEntryPointInput::new(
          entry_point=fixture_entry(digest, 12),
          function_symbol="embedding_prefill_4",
        ),
      ],
      fixture_file_limits(),
    )
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    let bundle = @artifact_file.load_admitted(
      authority,
      source,
      fixture_contracts(),
      fixture_file_limits(),
    )
    assert_eq(bundle.module_count(), 1)
    assert_false(authority.is_closed())
    authority.close()
  }
}

///|
async test "manifest and module snapshot digest mutations fail closed" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-file-digest")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let manifest = valid_manifest()
    write_file(root, "manifest.json", manifest)
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    let wrong_expected = expected_manifest(fixture_module_bytes())
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        wrong_expected,
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      DigestMismatch(ManifestFile) => ()
      _ => fail("expected manifest digest mismatch")
    } noraise {
      _ => fail("expected failure")
    }

    let mutated_manifest_array = manifest.to_array()
    mutated_manifest_array[0] = b'['
    let mutated_manifest = Bytes::from_array(mutated_manifest_array)
    replace_file(root, "manifest.json", mutated_manifest)
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(manifest),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      DigestMismatch(ManifestFile) => ()
      _ => fail("expected changed manifest rejection")
    } noraise {
      _ => fail("expected failure")
    }

    replace_file(root, "manifest.json", manifest)
    let mutated_module_array = fixture_module_bytes().to_array()
    mutated_module_array[0] = b'X'
    replace_file(root, "kernel.cubin", Bytes::from_array(mutated_module_array))
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(manifest),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      DigestMismatch(ModuleFile(0)) => ()
      _ => fail("expected changed module rejection")
    } noraise {
      _ => fail("expected failure")
    }

    replace_file(root, "kernel.cubin", fixture_module_bytes())
    let recovered = @artifact_file.load(
      authority,
      "manifest.json",
      expected_manifest(manifest),
      fixture_contracts(),
      fixture_file_limits(),
    )
    assert_eq(recovered.module_count(), 1)
    authority.close()
  }
}

///|
async test "symlink directory traversal absolute and aliases are rejected" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-file-path")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    write_file(root, "kernel.cubin", fixture_module_bytes())
    @fs.symlink(target="\{root}/kernel.cubin", "\{root}/kernel.link")
    @fs.mkdir("\{root}/kernel.dir")
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))

    let unsafe_paths = [
      ("kernel.link", false),
      ("kernel.dir", false),
      ("../kernel.cubin", true),
      ("/kernel.cubin", true),
    ]
    for index, test_case in unsafe_paths {
      let manifest = valid_manifest(path=test_case.0)
      replace_file(root, "manifest.json", manifest)
      try
        @artifact_file.load(
          authority,
          "manifest.json",
          expected_manifest(manifest),
          fixture_contracts(),
          fixture_file_limits(),
        )
      catch {
        InvalidPath(file=ModuleFile(0), issue=TraversalSegment | AbsolutePath) if test_case.1 =>
          ()
        UnsupportedPathType(ModuleFile(0)) if !test_case.1 => ()
        _ => fail("expected unsafe path rejection at \{index}")
      } noraise {
        _ => fail("expected failure")
      }
    }

    let duplicate_path_manifest = manifest_json([
      module_json(fixture_digest().sha256(), "kernel.cubin", [
        entry_json(1, 11, "embedding_prefill_2"),
      ]),
      module_json(fixture_digest().sha256(), "kernel.cubin", [
        entry_json(1, 12, "embedding_prefill_4"),
      ]),
    ])
    replace_file(root, "manifest.json", duplicate_path_manifest)
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(duplicate_path_manifest),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      InvalidPath(file=ModuleFile(1), issue=DuplicatePath) => ()
      _ => fail("expected duplicate path rejection")
    } noraise {
      _ => fail("expected failure")
    }
    authority.close()
  }
}

///|
async test "module per-file and aggregate limits fail before semantic admission" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-file-limits")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let manifest = valid_manifest()
    write_file(root, "manifest.json", manifest)
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(manifest),
        fixture_contracts(),
        fixture_file_limits(max_module_bytes=4L),
      )
    catch {
      FileTooLarge(file=ModuleFile(0), maximum=4L) => ()
      _ => fail("expected module file limit")
    } noraise {
      _ => fail("expected failure")
    }
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(manifest),
        fixture_contracts(),
        fixture_file_limits(max_total_module_bytes=4L),
      )
    catch {
      TotalModuleBytesExceeded(maximum=4L) => ()
      _ => fail("expected aggregate module limit")
    } noraise {
      _ => fail("expected failure")
    }
    authority.close()
  }
}

///|
async test "direct source aggregate preflight opens all files before snapshots" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-direct-preflight")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let corrupted = fixture_module_bytes().to_array()
    corrupted[0] = b'X'
    write_file(root, "one.cubin", Bytes::from_array(corrupted))
    write_file(root, "two.cubin", fixture_module_bytes())
    let digest = fixture_digest()
    let source = @artifact_file.KernelArtifactManifest::admit_files(
      [
        @artifact_file.KernelModuleFileInput::new(
          digest~,
          path_identifier="one.cubin",
        ),
        @artifact_file.KernelModuleFileInput::new(
          digest~,
          path_identifier="two.cubin",
        ),
      ],
      [],
      fixture_file_limits(),
    )
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))
    try
      @artifact_file.load_admitted(
        authority,
        source,
        fixture_contracts(),
        fixture_file_limits(
          max_total_module_bytes=fixture_module_bytes().length().to_int64(),
        ),
      )
    catch {
      TotalModuleBytesExceeded(maximum=actual) if actual ==
        fixture_module_bytes().length().to_int64() => ()
      _ => fail("unexpected direct aggregate preflight error")
    } noraise {
      _ => fail("aggregate-overflow source reached snapshot admission")
    }
    assert_false(authority.is_closed())
    authority.close()
  }
}

///|
async test "unknown and duplicate stable entry identities reach semantic authority" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="lunaflux-aot-file-entry")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    write_file(root, "kernel.cubin", fixture_module_bytes())
    let authority = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(root))

    let unknown = valid_manifest(entry_ids=[11, 99])
    write_file(root, "manifest.json", unknown)
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(unknown),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      ArtifactRejected(UnreferencedEntryPoint(identity)) =>
        assert_eq(identity.entry_point_id().as_int(), 99)
      _ => fail("expected unknown entry identity rejection")
    } noraise {
      _ => fail("expected failure")
    }
    let duplicate = valid_manifest(entry_ids=[11, 11])
    replace_file(root, "manifest.json", duplicate)
    try
      @artifact_file.load(
        authority,
        "manifest.json",
        expected_manifest(duplicate),
        fixture_contracts(),
        fixture_file_limits(),
      )
    catch {
      ArtifactRejected(DuplicateEntryPoint(identity)) =>
        assert_eq(identity.entry_point_id().as_int(), 11)
      _ => fail("expected duplicate entry identity rejection")
    } noraise {
      _ => fail("expected failure")
    }
    authority.close()
  }
}
