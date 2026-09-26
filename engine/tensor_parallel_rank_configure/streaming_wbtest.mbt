///|
test "rank configure v5 startup binds policy and rejects digest substitution" {
  let source = configure_test_source()
  let base = configure_test_startup(0, source)
  let capability = @streaming_ir.Capability::new(
    slots=4,
    host_bytes=1024L,
    page_bytes=256L,
    page_count=4096,
    reserve_pages=2,
    restore_ns=10L,
    recompute_ns=100L,
    separate_host_memory=true,
  )
  let startup = configure_test_startup(0, source, streaming=capability)
  let binding = configure_test_binding(0, startup)
  let contract = compose_rank_configure(
    binding,
    startup,
    source,
    configure_test_topology(),
    configure_test_envelope(0, startup),
  )
  let bytes = FixedArray::make(
    tensor_parallel_rank_configure_capacity(),
    b'\x00',
  )
  contract.copy_to(bytes, destination_offset=0)
  let decoded = decode_rank_configure(bytes, contract.length(), binding)
  assert_eq(decoded.startup().streaming(), Some(capability))
  assert_true(
    @tensor_parallel_worker_bootstrap.worker_contract_digest(startup) !=
    @tensor_parallel_worker_bootstrap.worker_contract_digest(base),
  )
  try
    compose_rank_configure(
      configure_test_binding(0, base),
      startup,
      source,
      configure_test_topology(),
      configure_test_envelope(0, base),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("policy substitution accepted")
  }
}
