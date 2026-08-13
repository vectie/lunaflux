///|
fn paged_exports(
  digest : @catalog.AotArtifactDigest,
) -> Array[@artifact.KernelEntryPointInput] {
  [
    export_input(digest, 11, 1, "shared_paged_symbol"),
    export_input(digest, 12, 2, "shared_paged_symbol"),
  ]
}

///|
test "full paged graph artifact admission includes non-KV entry points" {
  let source = b"paged-full-graph-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = paged_artifact_full_contracts(digest)
  let exports = [
    export_input(digest, 10, 3, "embedding_live"),
    export_input(digest, 11, 1, "shared_paged_symbol"),
    export_input(digest, 12, 2, "shared_paged_symbol"),
  ]
  let bundle = @artifact.admit_paged_kv(
    contracts,
    [@artifact.KernelModuleInput::new(digest~, module_bytes=source)],
    exports,
    limits(),
  )
  assert_eq(contracts.scope(), FullGraph)
  assert_eq(contracts.contracts().length(), 5)
  assert_eq(bundle.module_count(), 1)
  assert_eq(bundle.entry_point_count(), 3)
  assert_eq(bundle.entry_points()[0].function_symbol(), "embedding_live")
}

///|
test "paged artifact admission deduplicates shared layer identities" {
  let source = b"paged-multi-layer-cubin".to_fixedarray()
  let snapshot = Bytes::from_array(source)
  let digest = digest_for(source)
  let modules = [@artifact.KernelModuleInput::new(digest~, module_bytes=source)]
  let exports = paged_exports(digest)
  let contracts = paged_artifact_contracts(digest)
  assert_eq(contracts.contracts().length(), 4)
  assert_eq(
    contracts.contracts()[0].entry_point(),
    contracts.contracts()[2].entry_point(),
  )
  assert_eq(
    contracts.contracts()[1].entry_point(),
    contracts.contracts()[3].entry_point(),
  )
  let bundle = @artifact.admit_paged_kv(contracts, modules, exports, limits())
  source[0] = b'X'
  modules.push(modules[0])
  exports[0] = exports[1]
  assert_eq(bundle.model_identity(), contracts.model_identity())
  assert_eq(bundle.device_target(), contracts.device_target())
  assert_eq(bundle.catalog_version(), @catalog.CatalogVersion::v2())
  assert_eq(bundle.module_count(), 1)
  assert_eq(bundle.entry_point_count(), 2)
  assert_eq(bundle.modules()[0].module_bytes(), snapshot)
  assert_eq(bundle.entry_points()[0].identity(), entry(digest, 11, 1))
  assert_eq(bundle.entry_points()[1].identity(), entry(digest, 12, 2))
  assert_eq(bundle.entry_points()[0].function_symbol(), "shared_paged_symbol")
  assert_eq(bundle.entry_points()[1].function_symbol(), "shared_paged_symbol")
}

///|
test "paged artifact admission rejects missing and unreferenced inputs" {
  let source = b"paged-multi-layer-cubin".to_fixedarray()
  let digest = digest_for(source)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  let exports = paged_exports(digest)
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [],
      exports,
      limits(),
    )
  catch {
    @artifact.MissingModule(found) => assert_eq(found, digest)
    _ => fail("unexpected missing paged module error")
  } noraise {
    _ => fail("expected missing paged module")
  }
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [module_input],
      exports[:1],
      limits(),
    )
  catch {
    @artifact.MissingEntryPoint(found) => assert_eq(found, entry(digest, 12, 2))
    _ => fail("unexpected missing paged entry error")
  } noraise {
    _ => fail("expected missing paged entry")
  }
  let unused = export_input(digest, 12, 99, "unused_paged_symbol")
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [module_input],
      [exports[0], exports[1], unused],
      limits(),
    )
  catch {
    @artifact.UnreferencedEntryPoint(found) =>
      assert_eq(found, entry(digest, 12, 99))
    _ => fail("unexpected unreferenced paged entry error")
  } noraise {
    _ => fail("expected unreferenced paged entry")
  }
  let extra_source = b"extra-paged-module".to_fixedarray()
  let extra_digest = digest_for(extra_source)
  let extra = @artifact.KernelModuleInput::new(
    digest=extra_digest,
    module_bytes=extra_source,
  )
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [module_input, extra],
      exports,
      limits(),
    )
  catch {
    @artifact.UnreferencedModule(found) => assert_eq(found, extra_digest)
    _ => fail("unexpected unreferenced paged module error")
  } noraise {
    _ => fail("expected unreferenced paged module")
  }
}

///|
test "paged artifact admission rejects duplicate identities" {
  let source = b"paged-multi-layer-cubin".to_fixedarray()
  let digest = digest_for(source)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  let exports = paged_exports(digest)
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [module_input, module_input],
      exports,
      limits(),
    )
  catch {
    @artifact.DuplicateModule(found) => assert_eq(found, digest)
    _ => fail("unexpected duplicate paged module error")
  } noraise {
    _ => fail("expected duplicate paged module")
  }
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [module_input],
      [exports[0], exports[0], exports[1]],
      limits(),
    )
  catch {
    @artifact.DuplicateEntryPoint(found) =>
      assert_eq(found, entry(digest, 11, 1))
    _ => fail("unexpected duplicate paged entry error")
  } noraise {
    _ => fail("expected duplicate paged entry")
  }
}

///|
test "paged artifact digest validation remains exact" {
  let source = b"paged-multi-layer-cubin".to_fixedarray()
  let digest = digest_for(source)
  let wrong = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=b"mutated-paged-module".to_fixedarray(),
  )
  try
    @artifact.admit_paged_kv(
      paged_artifact_contracts(digest),
      [wrong],
      paged_exports(digest),
      limits(),
    )
  catch {
    @artifact.DigestMismatch(found) => assert_eq(found, digest)
    _ => fail("unexpected paged digest error")
  } noraise {
    _ => fail("expected paged digest mismatch")
  }
}
