///|
test "F32 reduction and gather preserve dtype in immutable sites" {
  let digest = collective_test_contract().digest()
  let sites = [
    TensorCollectiveSite::new(
      sequence=1,
      operation_id=0,
      kind=F32SumAllReduce,
      full_width=16,
      local_width=16,
    ),
    TensorCollectiveSite::new(
      sequence=2,
      operation_id=10,
      kind=F32AllGather,
      full_width=16,
      local_width=8,
    ),
  ]
  let contract = TensorCollectiveContract::new(
    digest~,
    rank=1,
    world_size=2,
    max_query_tokens=32,
    sites~,
  )
  assert_eq(contract.kinds[0], 3)
  assert_eq(contract.kinds[1], 4)
  assert_eq(
    authenticate_tensor_collective_claim(
      contract,
      1UL,
      1UL,
      1UL,
      0,
      1UL,
      1UL,
      1UL,
      1,
      0,
      F32SumAllReduce,
    ),
    0,
  )
  assert_collective_claim_rejected(
    contract,
    1UL,
    1UL,
    1UL,
    0,
    1UL,
    1UL,
    1UL,
    1,
    0,
    Bf16SumAllReduce,
  )
  assert_eq(nccl_collective_kind(F32SumAllReduce), @nccl.AllReduce)
  assert_eq(nccl_collective_kind(F32AllGather), @nccl.AllGather)
}
