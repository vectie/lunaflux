///|
fn expect_paged_v4_missing_module(
  contracts : @launch_contract.PagedV4LaunchContractSet,
  exports : ArrayView[@artifact.KernelEntryPointInput],
  digest : @catalog.AotArtifactDigest,
) -> Unit raise {
  try @artifact.admit_paged_v4(contracts, [], exports, limits()) catch {
    MissingModule(found) => assert_eq(found, digest)
    _ => fail("unexpected catalog-v4 missing-module error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted a missing module")
  }
}

///|
test "catalog v4 artifact admission preserves exact order and ownership" {
  let source = b"paged-v4-full-graph-cubin".to_fixedarray()
  let snapshot = Bytes::from_array(source)
  let digest = digest_for(source)
  let contracts = paged_artifact_v4_contracts(digest)
  let modules = [@artifact.KernelModuleInput::new(digest~, module_bytes=source)]
  let exports = paged_artifact_v4_exports(digest)
  let bundle = @artifact.admit_paged_v4(contracts, modules, exports, limits())
  source[0] = b'X'
  modules.push(modules[0])
  exports[0] = exports[1]
  assert_eq(bundle.model_identity(), contracts.model_identity())
  assert_eq(bundle.device_target(), contracts.device_target())
  assert_eq(bundle.catalog_version(), @catalog.CatalogVersion::v4())
  assert_eq(bundle.module_count(), 1)
  assert_eq(bundle.entry_point_count(), 3)
  assert_eq(bundle.modules()[0].module_bytes(), snapshot)
  assert_eq(bundle.entry_points()[0].identity(), entry(digest, 10, 3))
  assert_eq(bundle.entry_points()[1].identity(), entry(digest, 11, 1))
  assert_eq(bundle.entry_points()[2].identity(), entry(digest, 12, 2))
}

///|
test "catalog v4 artifact admission rejects missing and extra authority" {
  let source = b"paged-v4-full-graph-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = paged_artifact_v4_contracts(digest)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  let exports = paged_artifact_v4_exports(digest)
  expect_paged_v4_missing_module(contracts, exports, digest)
  try
    @artifact.admit_paged_v4(contracts, [module_input], exports[:2], limits())
  catch {
    MissingEntryPoint(found) => assert_eq(found, entry(digest, 12, 2))
    _ => fail("unexpected catalog-v4 missing-entry error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted a missing entry")
  }
  let unused = export_input(digest, 12, 99, "unused_v4")
  try
    @artifact.admit_paged_v4(
      contracts,
      [module_input],
      [exports[0], exports[1], exports[2], unused],
      limits(),
    )
  catch {
    UnreferencedEntryPoint(found) => assert_eq(found, entry(digest, 12, 99))
    _ => fail("unexpected catalog-v4 unreferenced-entry error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted an extra entry")
  }
  let extra_bytes = b"paged-v4-extra-module".to_fixedarray()
  let extra_digest = digest_for(extra_bytes)
  let extra = @artifact.KernelModuleInput::new(
    digest=extra_digest,
    module_bytes=extra_bytes,
  )
  try
    @artifact.admit_paged_v4(
      contracts,
      [module_input, extra],
      exports,
      limits(),
    )
  catch {
    UnreferencedModule(found) => assert_eq(found, extra_digest)
    _ => fail("unexpected catalog-v4 unreferenced-module error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted an extra module")
  }
}

///|
test "catalog v4 artifact admission rejects duplicate authority" {
  let source = b"paged-v4-full-graph-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = paged_artifact_v4_contracts(digest)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  let exports = paged_artifact_v4_exports(digest)
  try
    @artifact.admit_paged_v4(
      contracts,
      [module_input, module_input],
      exports,
      limits(),
    )
  catch {
    DuplicateModule(found) => assert_eq(found, digest)
    _ => fail("unexpected catalog-v4 duplicate-module error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted a duplicate module")
  }
  try
    @artifact.admit_paged_v4(
      contracts,
      [module_input],
      [exports[0], exports[0], exports[1], exports[2]],
      limits(),
    )
  catch {
    DuplicateEntryPoint(found) => assert_eq(found, entry(digest, 10, 3))
    _ => fail("unexpected catalog-v4 duplicate-entry error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted a duplicate entry")
  }
}

///|
test "catalog v4 artifact admission hashes bytes and validates symbols" {
  let source = b"paged-v4-full-graph-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = paged_artifact_v4_contracts(digest)
  let exports = paged_artifact_v4_exports(digest)
  let wrong = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=b"wrong-paged-v4-cubin".to_fixedarray(),
  )
  try @artifact.admit_paged_v4(contracts, [wrong], exports, limits()) catch {
    DigestMismatch(found) => assert_eq(found, digest)
    _ => fail("unexpected catalog-v4 digest error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted the wrong digest")
  }
  let invalid = @artifact.KernelEntryPointInput::new(
    entry_point=entry(digest, 11, 1),
    function_symbol="bad-symbol",
  )
  try
    @artifact.admit_paged_v4(
      contracts,
      [@artifact.KernelModuleInput::new(digest~, module_bytes=source)],
      [exports[0], invalid, exports[2]],
      limits(),
    )
  catch {
    InvalidFunctionSymbol(rule=InvalidContinuation, ..) => ()
    _ => fail("unexpected catalog-v4 symbol error")
  } noraise {
    _ => fail("catalog-v4 artifact admission accepted an invalid symbol")
  }
}
