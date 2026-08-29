///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn copy_digest_into(
  destination : FixedArray[Byte],
  offset : Int,
  value : String,
) -> Unit raise RankGroupWireError {
  if !is_lower_sha256(value) {
    raise Invalid(Digest)
  }
  let encoded = @utf8.encode(value)
  for index in 0..<64 {
    destination[offset + index] = encoded[index]
  }
}

///|
pub fn RankGroupWireBinding::new(
  identity~ : @spec.ModelIdentity,
  model_generation_value~ : UInt64,
  group_generation_value~ : UInt64,
  rank~ : Int,
  world_size~ : Int,
  group_digest~ : String,
  topology_digest~ : String,
  worker_digest~ : String,
  device_plan_digest~ : String,
  execution_plan_digest~ : String,
  collective_digest~ : String,
  kv_digest~ : String,
) -> RankGroupWireBinding raise RankGroupWireError {
  if model_generation_value == 0UL {
    raise Invalid(ModelGeneration)
  }
  if group_generation_value == 0UL {
    raise Invalid(GroupGeneration)
  }
  if world_size < 2 || world_size > RANK_GROUP_WIRE_MAX_WORLD_SIZE {
    raise Invalid(WorldSize)
  }
  if rank < 0 || rank >= world_size {
    raise Invalid(Rank)
  }
  let canonical_digests = FixedArray::make(576, b'\x00')
  copy_digest_into(canonical_digests, 0, identity.content().as_hex())
  copy_digest_into(canonical_digests, 64, identity.plan().as_hex())
  copy_digest_into(canonical_digests, 128, group_digest)
  copy_digest_into(canonical_digests, 192, topology_digest)
  copy_digest_into(canonical_digests, 256, worker_digest)
  copy_digest_into(canonical_digests, 320, device_plan_digest)
  copy_digest_into(canonical_digests, 384, execution_plan_digest)
  copy_digest_into(canonical_digests, 448, collective_digest)
  copy_digest_into(canonical_digests, 512, kv_digest)
  {
    identity,
    model_generation_value,
    group_generation_value,
    rank,
    world_size,
    group_digest,
    topology_digest,
    worker_digest,
    device_plan_digest,
    execution_plan_digest,
    collective_digest,
    kv_digest,
    canonical_digests,
  }
}

///|
pub fn RankGroupWireBinding::identity(
  self : RankGroupWireBinding,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn RankGroupWireBinding::model_generation_value(
  self : RankGroupWireBinding,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn RankGroupWireBinding::group_generation_value(
  self : RankGroupWireBinding,
) -> UInt64 {
  self.group_generation_value
}

///|
pub fn RankGroupWireBinding::rank(self : RankGroupWireBinding) -> Int {
  self.rank
}

///|
pub fn RankGroupWireBinding::world_size(self : RankGroupWireBinding) -> Int {
  self.world_size
}

///|
pub fn RankGroupWireBinding::group_digest(
  self : RankGroupWireBinding,
) -> String {
  self.group_digest
}

///|
pub fn RankGroupWireBinding::topology_digest(
  self : RankGroupWireBinding,
) -> String {
  self.topology_digest
}

///|
pub fn RankGroupWireBinding::worker_digest(
  self : RankGroupWireBinding,
) -> String {
  self.worker_digest
}

///|
pub fn RankGroupWireBinding::device_plan_digest(
  self : RankGroupWireBinding,
) -> String {
  self.device_plan_digest
}

///|
pub fn RankGroupWireBinding::execution_plan_digest(
  self : RankGroupWireBinding,
) -> String {
  self.execution_plan_digest
}

///|
pub fn RankGroupWireBinding::collective_digest(
  self : RankGroupWireBinding,
) -> String {
  self.collective_digest
}

///|
pub fn RankGroupWireBinding::kv_digest(self : RankGroupWireBinding) -> String {
  self.kv_digest
}

///|
fn RankGroupWireBinding::same_group_as(
  self : RankGroupWireBinding,
  other : RankGroupWireBinding,
) -> Bool {
  if self.model_generation_value != other.model_generation_value ||
    self.group_generation_value != other.group_generation_value ||
    self.world_size != other.world_size {
    return false
  }
  // Model content, model plan, group bootstrap, and topology are shared.
  for index in 0..<256 {
    if self.canonical_digests[index] != other.canonical_digests[index] {
      return false
    }
  }
  true
}

///|
fn RankGroupWireBinding::exactly_matches(
  self : RankGroupWireBinding,
  other : RankGroupWireBinding,
) -> Bool {
  if self.rank != other.rank ||
    self.world_size != other.world_size ||
    self.model_generation_value != other.model_generation_value ||
    self.group_generation_value != other.group_generation_value {
    return false
  }
  for index in 0..<self.canonical_digests.length() {
    if self.canonical_digests[index] != other.canonical_digests[index] {
      return false
    }
  }
  true
}
