///|
test "compressed radix sizing is exact and constructor revalidates limits" {
  let limits = PrefixIndexLimits::new(
    entry_capacity=8,
    node_capacity=32,
    token_capacity=32,
    max_tokens_per_entry=16,
    page_capacity=16,
    max_scope_bytes=64,
    max_active_references_per_entry=4,
  )
  assert_eq(PrefixIndex::required_int_cells(limits), 1192UL)
  assert_eq(PrefixIndex::required_bool_cells(limits), 136UL)
  assert_eq(PrefixIndex::required_byte_cells(limits), 576UL)
  assert_eq(PrefixIndex::required_reference_cells(limits), 24UL)
  ignore(PrefixIndex::new(limits))

  let forged : PrefixIndexLimits = {
    entry_capacity: 8,
    node_capacity: 32,
    token_capacity: 32,
    max_tokens_per_entry: 16,
    page_capacity: 16,
    max_scope_bytes: 0,
    max_active_references_per_entry: 4,
    max_generation: 100,
  }
  try PrefixIndex::new(forged) catch {
    InvalidLimit(ScopeBytes) => ()
    _ => fail("unexpected forged-limit constructor error")
  } noraise {
    _ => fail("forged prefix limits reached allocation")
  }
}

///|
test "aggregate scope arithmetic rejects its exact checked boundary" {
  ignore(
    PrefixIndexLimits::new(
      entry_capacity=65536,
      node_capacity=65536,
      token_capacity=65536,
      max_tokens_per_entry=1,
      page_capacity=1,
      max_scope_bytes=256,
      max_active_references_per_entry=1,
    ),
  )
  try
    PrefixIndexLimits::new(
      entry_capacity=65536,
      node_capacity=65536,
      token_capacity=65536,
      max_tokens_per_entry=1,
      page_capacity=1,
      max_scope_bytes=257,
      max_active_references_per_entry=1,
    )
  catch {
    InvalidLimit(ScopeBytes) => ()
    _ => fail("unexpected aggregate scope error")
  } noraise {
    _ => fail("oversized aggregate scope was accepted")
  }
}

///|
test "compressed arena relations reject unreachable declared envelopes" {
  try
    PrefixIndexLimits::new(
      entry_capacity=3,
      node_capacity=2,
      token_capacity=4,
      max_tokens_per_entry=4,
      page_capacity=2,
      max_scope_bytes=4,
      max_active_references_per_entry=1,
    )
  catch {
    InvalidLimit(NodeCapacity) => ()
    _ => fail("unexpected terminal-node envelope error")
  } noraise {
    _ => fail("entry capacity exceeded the terminal-node arena")
  }
  try
    PrefixIndexLimits::new(
      entry_capacity=1,
      node_capacity=2,
      token_capacity=3,
      max_tokens_per_entry=4,
      page_capacity=2,
      max_scope_bytes=4,
      max_active_references_per_entry=1,
    )
  catch {
    InvalidLimit(TokenCapacity) => ()
    _ => fail("unexpected token-arena envelope error")
  } noraise {
    _ => fail("one entry exceeded the aggregate token-cell arena")
  }
}
