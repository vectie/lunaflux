///|
priv struct ContractEncoder {
  bytes : Array[Byte]
}

///|
fn ContractEncoder::append_u64(self : ContractEncoder, value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    self.bytes.push(byte)
  }
}

///|
fn ContractEncoder::append_int(self : ContractEncoder, value : Int) -> Unit {
  self.append_u64(value.to_int64().reinterpret_as_uint64())
}

///|
fn ContractEncoder::append_string(
  self : ContractEncoder,
  value : StringView,
) -> Unit {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.bytes.push(byte)
  }
}

///|
fn map_kind(
  kind : @tensor_parallel_plan.TensorCollectiveKind,
) -> @device.TensorCollectiveKind {
  match kind {
    SumAllReduce => Bf16SumAllReduce
    LastAxisAllGather => Bf16AllGather
  }
}

///|
/// Authenticates one immutable architecture-neutral rank plan and projects
/// only its fixed scalar collective contract into the device package.
pub fn authenticate_rank_plan(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  expected_identity~ : @spec.ModelIdentity,
  max_query_tokens~ : Int,
) -> @device.TensorCollectiveContract raise TensorParallelCollectiveError {
  if rank_plan.identity() != expected_identity {
    raise InvalidRankPlan(RankPlanIdentity)
  }
  let topology = rank_plan.topology()
  let rank = topology.rank()
  let world_size = topology.world_size()
  let count = rank_plan.collective_count()
  if world_size < 2 ||
    world_size > 16 ||
    rank < 0 ||
    rank >= world_size ||
    count == 0 ||
    max_query_tokens <= 0 {
    raise InvalidRankPlan(RankPlanShape)
  }
  let encoder : ContractEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-collective-contract.v1")
  encoder.append_string(expected_identity.content().as_hex())
  encoder.append_string(expected_identity.plan().as_hex())
  encoder.append_int(rank)
  encoder.append_int(world_size)
  encoder.append_int(max_query_tokens)
  encoder.append_int(count)
  let sites : Array[@device.TensorCollectiveSite] = []
  for index in 0..<count {
    let site = rank_plan.collective(index).unwrap()
    let sequence = site.sequence().as_int()
    let operation_id = site.operation().as_int()
    let kind = map_kind(site.kind())
    let full_width = site.full_width()
    let local_width = site.local_width()
    encoder.append_int(sequence)
    encoder.append_int(operation_id)
    encoder.append_int(
      match kind {
        Bf16SumAllReduce => 1
        Bf16AllGather => 2
      },
    )
    encoder.append_int(full_width)
    encoder.append_int(local_width)
    let projected = @device.TensorCollectiveSite::new(
      sequence~,
      operation_id~,
      kind~,
      full_width~,
      local_width~,
    ) catch {
      error => raise Device(DeviceContract, error)
    }
    sites.push(projected)
  }
  let digest_hex = @crypto.bytes_to_hex_string(
    @crypto.sha256(Bytes::from_array(encoder.bytes)),
  )
  let digest = @device.TensorCollectiveContractDigest::from_sha256(digest_hex) catch {
    error => raise Device(CanonicalContract, error)
  }
  @device.TensorCollectiveContract::new(
    digest~,
    rank~,
    world_size~,
    max_query_tokens~,
    sites~,
  ) catch {
    error => raise Device(DeviceContract, error)
  }
}
