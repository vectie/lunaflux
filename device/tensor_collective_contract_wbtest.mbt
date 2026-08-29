///|
fn collective_test_contract() -> TensorCollectiveContract {
  let sites = [
    try! TensorCollectiveSite::new(
      sequence=1,
      operation_id=0,
      kind=Bf16SumAllReduce,
      full_width=16,
      local_width=16,
    ),
    try! TensorCollectiveSite::new(
      sequence=2,
      operation_id=10,
      kind=Bf16AllGather,
      full_width=16,
      local_width=8,
    ),
  ]
  try! TensorCollectiveContract::new(
    digest={
      hex: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
    rank=1,
    world_size=2,
    max_query_tokens=32,
    sites~,
  )
}

///|
fn assert_collective_claim_rejected(
  contract : TensorCollectiveContract,
  generation : UInt64,
  current_plan_sequence : UInt64,
  next_global : UInt64,
  next_site : Int,
  claimed_generation : UInt64,
  claimed_plan_sequence : UInt64,
  claimed_global_sequence : UInt64,
  claimed_site_sequence : Int,
  claimed_operation_id : Int,
  claimed_kind : TensorCollectiveKind,
) -> Unit raise {
  try
    authenticate_tensor_collective_claim(
      contract, generation, current_plan_sequence, next_global, next_site, claimed_generation,
      claimed_plan_sequence, claimed_global_sequence, claimed_site_sequence, claimed_operation_id,
      claimed_kind,
    )
  catch {
    InvalidArgument(TensorCollectiveExecute) => ()
    _ => fail("unexpected collective authentication failure")
  } noraise {
    _ => fail("substituted collective claim was admitted")
  }
}

///|
fn assert_collective_elements_rejected(
  tokens : Int64,
  width : Int64,
) -> Unit raise {
  try checked_tensor_collective_elements(tokens, width) catch {
    InvalidArgument(TensorCollectiveExecute) => ()
    _ => fail("unexpected checked-element failure")
  } noraise {
    _ => fail("invalid collective element product was admitted")
  }
}

///|
test "generic contract rejects malformed digests sites and widths" {
  for
    digest in [
      "short", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ] {
    try TensorCollectiveContractDigest::from_sha256(digest) catch {
      InvalidArgument(TensorCollectiveContractCreate) => ()
      _ => fail("unexpected digest error")
    } noraise {
      _ => fail("malformed contract digest was admitted")
    }
  }
  let digest = collective_test_contract().digest()
  let bad_sites = [
    try! TensorCollectiveSite::new(
      sequence=2,
      operation_id=1,
      kind=Bf16AllGather,
      full_width=15,
      local_width=8,
    ),
  ]
  try
    TensorCollectiveContract::new(
      digest~,
      rank=0,
      world_size=2,
      max_query_tokens=32,
      sites=bad_sites,
    )
  catch {
    InvalidArgument(TensorCollectiveContractCreate) => ()
    _ => fail("unexpected site-table error")
  } noraise {
    _ => fail("malformed site table was admitted")
  }
}

///|
test "query-token bounds and checked collective element multiplication" {
  let contract = collective_test_contract()
  assert_eq(contract.max_query_tokens(), 32)
  require_tensor_collective_query_tokens(contract, 1)
  require_tensor_collective_query_tokens(contract, 32)
  for tokens in [0, 33] {
    try require_tensor_collective_query_tokens(contract, tokens) catch {
      InvalidArgument(TensorCollectiveExecute) => ()
      _ => fail("unexpected live-query-token failure")
    } noraise {
      _ => fail("out-of-envelope live query tokens were admitted")
    }
  }
  assert_eq(checked_tensor_collective_elements(3L, 16L), 48L)
  // Three rank-major query rows: local shard send and full-vocabulary receive.
  assert_eq(checked_tensor_collective_elements(3L, 8L), 24L)
  assert_collective_elements_rejected(0L, 16L)
  assert_collective_elements_rejected(1L, 0L)
  assert_collective_elements_rejected(9223372036854775807L, 2L)
  let digest = collective_test_contract().digest()
  let site = try! TensorCollectiveSite::new(
    sequence=1,
    operation_id=0,
    kind=Bf16SumAllReduce,
    full_width=16,
    local_width=16,
  )
  try
    TensorCollectiveContract::new(
      digest~,
      rank=0,
      world_size=2,
      max_query_tokens=0,
      sites=[site],
    )
  catch {
    InvalidArgument(TensorCollectiveContractCreate) => ()
    _ => fail("unexpected zero query-token contract failure")
  } noraise {
    _ => fail("zero maximum query tokens were admitted")
  }
}

///|
test "collective runtime rejects pre-2.14.3 admission ranges" {
  try
    admit_tensor_collective_runtime(
      minimum_version_code=21402,
      maximum_version_code=22999,
    )
  catch {
    InvalidArgument(TensorCollectiveRuntimeAdmission) => ()
    _ => fail("unexpected BF16 runtime-range error")
  } noraise {
    _ => fail("pre-nonblocking NCCL range was admitted")
  }
}

///|
test "launch authentication rejects identity order and topology substitution" {
  let contract = collective_test_contract()
  assert_eq(
    authenticate_tensor_collective_claim(
      contract,
      7UL,
      0UL,
      1UL,
      2,
      7UL,
      1UL,
      1UL,
      1,
      0,
      Bf16SumAllReduce,
    ),
    0,
  )
  assert_eq(
    authenticate_tensor_collective_claim(
      contract,
      7UL,
      1UL,
      2UL,
      1,
      7UL,
      1UL,
      2UL,
      2,
      10,
      Bf16AllGather,
    ),
    1,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    8UL,
    1UL,
    1UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    7UL,
    1UL,
    2UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    7UL,
    1UL,
    1UL,
    2,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    7UL,
    1UL,
    1UL,
    1,
    1,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    7UL,
    1UL,
    1UL,
    1,
    0,
    Bf16AllGather,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    1UL,
    2UL,
    1,
    7UL,
    1UL,
    1UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    0UL,
    1UL,
    2,
    7UL,
    1UL,
    1UL,
    2,
    10,
    Bf16AllGather,
  )
}

///|
test "first and later plans require the exact predecessor plus one" {
  let contract = collective_test_contract()
  assert_collective_claim_rejected(
    contract,
    7UL,
    5UL,
    1UL,
    2,
    7UL,
    5UL,
    1UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    5UL,
    1UL,
    2,
    7UL,
    7UL,
    1UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_eq(
    authenticate_tensor_collective_claim(
      contract,
      7UL,
      5UL,
      1UL,
      2,
      7UL,
      6UL,
      1UL,
      1,
      0,
      Bf16SumAllReduce,
    ),
    0,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    6UL,
    2UL,
    1,
    7UL,
    7UL,
    2UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    6UL,
    3UL,
    2,
    7UL,
    6UL,
    3UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_collective_claim_rejected(
    contract,
    7UL,
    6UL,
    3UL,
    2,
    7UL,
    7UL,
    3UL,
    2,
    10,
    Bf16AllGather,
  )
}

///|
test "collective scalar authentication release-allocation positive control" {
  let contract = collective_test_contract()
  let allocated = Bytes::make(contract.site_count() + 17, b'M')
  assert_eq(allocated.length(), 19)
  for global in 1..<=4096 {
    let plan = global.to_uint64()
    assert_eq(
      authenticate_tensor_collective_claim(
        contract,
        7UL,
        plan - 1UL,
        plan,
        2,
        7UL,
        plan,
        plan,
        1,
        0,
        Bf16SumAllReduce,
      ),
      0,
    )
  }
}
