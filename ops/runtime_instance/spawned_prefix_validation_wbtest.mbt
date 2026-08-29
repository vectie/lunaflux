///|
fn wb_spawned_prefix_contract(
  first : FixedArray[Byte],
  second : FixedArray[Byte],
) -> RuntimeSpawnedPrefixValidationContract raise {
  RuntimeSpawnedPrefixValidationContract::new(
    first,
    second,
    spawned_validation_wb_identity(),
    1031,
    2049,
    8,
    9,
    8,
    16,
    8,
    1024,
  )
}

///|
test "spawned prefix contract owns exact full-page then private-tail geometry" {
  let first : FixedArray[Byte] = [b'a', b'b']
  let second : FixedArray[Byte] = [b'c', b'd']
  let contract = wb_spawned_prefix_contract(first, second)
  first[0] = b'z'
  second[0] = b'y'
  assert_eq(contract.first_frame[0], b'a')
  assert_eq(contract.second_frame[0], b'c')

  let invalid = FixedArray::make(1, b'x')
  try
    RuntimeSpawnedPrefixValidationContract::new(
      invalid,
      invalid,
      spawned_validation_wb_identity(),
      1,
      2,
      7,
      8,
      7,
      16,
      8,
      16,
    )
  catch {
    RuntimeSpawnedPrefixValidationFailed(SpawnedPrefixContract) => ()
    _ => abort("unexpected unaligned prefix contract error")
  } noraise {
    _ => abort("unaligned first prompt passed prefix contract")
  }
  try
    RuntimeSpawnedPrefixValidationContract::new(
      invalid,
      invalid,
      spawned_validation_wb_identity(),
      1,
      2,
      8,
      10,
      8,
      16,
      8,
      16,
    )
  catch {
    RuntimeSpawnedPrefixValidationFailed(SpawnedPrefixContract) => ()
    _ => abort("unexpected private-tail contract error")
  } noraise {
    _ => abort("multi-token private tail passed exact prefix contract")
  }
}

///|
fn wb_spawned_prefix_terminal_sample(
  hits : UInt64,
  misses : UInt64,
  publications : UInt64,
  reused : UInt64,
) -> SpawnedPrefixTelemetrySample {
  {
    queue_depth: 0,
    active_requests: 0,
    kv_pages_used: 1,
    kv_pages_free: 31,
    prefix_lookups: 2UL,
    prefix_hits: hits,
    prefix_misses: misses,
    prefix_publications: publications,
    prefix_tokens_reused: reused,
    prefix_tokens_computed: 9UL,
    prefix_entries: 1,
    prefix_pages: 1,
  }
}

///|
test "spawned prefix evidence rejects disabled readonly seed and scope miss" {
  assert_true(
    spawned_prefix_after_second_valid(
      wb_spawned_prefix_terminal_sample(1UL, 1UL, 1UL, 8UL),
      32,
      8,
      9,
      8,
      1,
    ),
  )
  // Disabled lookup cannot prove two authenticated requests.
  let disabled = wb_spawned_prefix_terminal_sample(0UL, 1UL, 0UL, 0UL)
  assert_false(spawned_prefix_after_second_valid(disabled, 32, 8, 9, 8, 1))
  // A read-only first request cannot publish the reusable page.
  let read_only_seed = wb_spawned_prefix_terminal_sample(0UL, 2UL, 0UL, 0UL)
  assert_false(
    spawned_prefix_after_second_valid(read_only_seed, 32, 8, 9, 8, 1),
  )
  // Same prompt bytes in another authenticated scope remain a miss.
  let scope_mismatch = wb_spawned_prefix_terminal_sample(0UL, 2UL, 1UL, 0UL)
  assert_false(
    spawned_prefix_after_second_valid(scope_mismatch, 32, 8, 9, 8, 1),
  )
}
