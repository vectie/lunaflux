///|
test "tensor-parallel artifact admits unique rank-local identities once" {
  let source = b"tensor-parallel-shared-cubin".to_fixedarray()
  let snapshot = Bytes::from_array(source)
  let digest = digest_for(source)
  let contracts = tp_contracts(digest)
  let exports = tp_export_inputs(contracts)
  assert_eq(contracts.profiles().length(), 2)
  assert_eq(contracts.contracts().length(), 24)
  assert_eq(exports.length(), 9)
  let modules = [@artifact.KernelModuleInput::new(digest~, module_bytes=source)]
  let bundle = @artifact.admit_tensor_parallel(
    contracts,
    modules,
    exports,
    limits(max_entry_points=16),
  )
  source[0] = b'X'
  modules.push(modules[0])
  exports[0] = exports[1]
  assert_eq(bundle.model_identity(), contracts.model_identity())
  assert_eq(bundle.device_target(), contracts.device_target())
  assert_eq(bundle.catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(bundle.module_count(), 1)
  assert_eq(bundle.entry_point_count(), 9)
  assert_eq(bundle.total_module_bytes(), snapshot.length().to_int64())
  assert_eq(bundle.modules()[0].module_bytes(), snapshot)
  for admitted in bundle.entry_points() {
    assert_eq(admitted.module_index(), 0)
    assert_eq(admitted.function_symbol(), "shared_tp_symbol")
  }
  for contract in contracts.contracts() {
    assert_true(bundle.entry_point(contract.entry_point()) is Some(_))
  }
}

///|
test "tensor-parallel artifact rejects missing and extra inputs" {
  let source = b"tensor-parallel-shared-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = tp_contracts(digest)
  let exports = tp_export_inputs(contracts)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [],
      exports,
      limits(max_entry_points=16),
    )
  catch {
    MissingModule(found) => assert_eq(found, digest)
    _ => fail("unexpected missing TP module error")
  } noraise {
    _ => fail("expected missing TP module")
  }
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [module_input],
      exports[:8],
      limits(max_entry_points=16),
    )
  catch {
    MissingEntryPoint(found) => assert_eq(found, exports[8].identity())
    _ => fail("unexpected missing TP entry-point error")
  } noraise {
    _ => fail("expected missing TP entry point")
  }
  let extra_source = b"foreign-tensor-parallel-cubin".to_fixedarray()
  let extra_digest = digest_for(extra_source)
  let extra_module = @artifact.KernelModuleInput::new(
    digest=extra_digest,
    module_bytes=extra_source,
  )
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [module_input, extra_module],
      exports,
      limits(max_entry_points=16),
    )
  catch {
    UnreferencedModule(found) => assert_eq(found, extra_digest)
    _ => fail("unexpected extra TP module error")
  } noraise {
    _ => fail("expected unreferenced TP module")
  }
  let foreign = @artifact.KernelEntryPointInput::new(
    entry_point=entry(extra_digest, 99, 199),
    function_symbol="foreign_tp_symbol",
  )
  let with_foreign = exports.copy()
  with_foreign.push(foreign)
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [module_input],
      with_foreign,
      limits(max_entry_points=16),
    )
  catch {
    UnreferencedEntryPoint(found) =>
      assert_eq(found, entry(extra_digest, 99, 199))
    _ => fail("unexpected foreign TP entry-point error")
  } noraise {
    _ => fail("expected foreign TP entry-point rejection")
  }
}

///|
test "tensor-parallel artifact retains duplicate and digest rejection" {
  let source = b"tensor-parallel-shared-cubin".to_fixedarray()
  let digest = digest_for(source)
  let contracts = tp_contracts(digest)
  let exports = tp_export_inputs(contracts)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=source,
  )
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [module_input, module_input],
      exports,
      limits(max_entry_points=16),
    )
  catch {
    DuplicateModule(found) => assert_eq(found, digest)
    _ => fail("unexpected duplicate TP module error")
  } noraise {
    _ => fail("expected duplicate TP module")
  }
  let duplicate_exports = exports.copy()
  duplicate_exports.push(exports[0])
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [module_input],
      duplicate_exports,
      limits(max_entry_points=16),
    )
  catch {
    DuplicateEntryPoint(found) => assert_eq(found, exports[0].identity())
    _ => fail("unexpected duplicate TP entry-point error")
  } noraise {
    _ => fail("expected duplicate TP entry point")
  }
  let wrong = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=b"mutated-tensor-parallel-cubin".to_fixedarray(),
  )
  try
    @artifact.admit_tensor_parallel(
      contracts,
      [wrong],
      exports,
      limits(max_entry_points=16),
    )
  catch {
    DigestMismatch(found) => assert_eq(found, digest)
    _ => fail("unexpected TP digest mismatch error")
  } noraise {
    _ => fail("expected TP digest mismatch")
  }
}
