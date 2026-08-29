///|
fn bootstrap_test_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "2222222222222222222222222222222222222222222222222222222222222222",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn bootstrap_test_rendezvous(
  fill? : Byte = b'\x5a',
) -> @device.TensorCollectiveGroupId {
  try! @device.tensor_collective_group_id_from_bytes(Bytes::make(128, fill))
}

///|
fn bootstrap_test_claim(
  rank : Int,
  world_size? : Int = 2,
  kv_hex? : String = "6666666666666666666666666666666666666666666666666666666666666666",
  execution_hex? : String = "8888888888888888888888888888888888888888888888888888888888888888",
) -> RankBootstrapClaim {
  bootstrap_test_claim_for(
    bootstrap_test_identity(),
    rank,
    world_size,
    kv_hex,
    execution_hex,
  )
}

///|
fn bootstrap_test_claim_for(
  identity : @spec.ModelIdentity,
  rank : Int,
  world_size : Int,
  kv_hex : String,
  execution_hex : String,
) -> RankBootstrapClaim {
  {
    identity,
    model_generation_value: 7UL,
    group_generation_value: 9UL,
    rank,
    world_size,
    device_ordinal: 4 + rank,
    worker_digest: {
      hex: if rank == 0 {
        "3333333333333333333333333333333333333333333333333333333333333333"
      } else {
        "3434343434343434343434343434343434343434343434343434343434343434"
      },
    },
    device_plan_digest: {
      hex: if rank == 0 {
        "4444444444444444444444444444444444444444444444444444444444444444"
      } else {
        "4545454545454545454545454545454545454545454545454545454545454545"
      },
    },
    execution_plan_digest_sha256: execution_hex,
    collective_digest: try! @device.TensorCollectiveContractDigest::from_sha256(
      if rank == 0 {
        "5555555555555555555555555555555555555555555555555555555555555555"
      } else {
        "5656565656565656565656565656565656565656565656565656565656565656"
      },
    ),
    kv_digest: { hex: kv_hex },
  }
}

///|
fn bootstrap_test_group(
  claims? : ArrayView[RankBootstrapClaim] = [
    bootstrap_test_claim(0),
    bootstrap_test_claim(1),
  ],
  world_size? : Int = 2,
) -> TensorParallelGroupBootstrap raise TensorParallelWorkerBootstrapError {
  admit_claims(
    bootstrap_test_identity(),
    7UL,
    9UL,
    { hex: "7777777777777777777777777777777777777777777777777777777777777777" },
    21403,
    bootstrap_test_rendezvous(),
    world_size,
    claims,
  )
}
