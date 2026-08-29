///|
/// Untrusted, authority-free deployment claim carried by a v5 source frame.
/// Admission against independently probed topology is required before use.
pub struct TensorParallelRankRuntimeClaim {
  priv rank : Int
  priv device_ordinal : Int
  priv memory_ceiling_bytes : Int64
  priv weight_reserved_bytes : Int64
  priv activation_workspace_reserved_bytes : Int64
  priv kv_reserved_bytes : Int64
  priv collective_reserved_bytes : Int64
} derive(Debug, Eq)

///|
/// Canonical rank-ordered deployment policy. It contains only untrusted scalar
/// claims and grants no device, memory, collective, or filesystem authority.
pub struct TensorParallelBootstrapPolicy {
  priv ranks : ReadOnlyArray[TensorParallelRankRuntimeClaim]
  priv minimum_collective_runtime_version_code : Int
  priv maximum_collective_runtime_version_code : Int
  priv collective_exchange_timeout_millis : Int
} derive(Debug, Eq)

///|
pub fn EncodedBootstrapSource::tensor_parallel_policy(
  self : EncodedBootstrapSource,
) -> TensorParallelBootstrapPolicy? {
  self.tensor_parallel_policy
}

///|
pub fn TensorParallelRankRuntimeClaim::new(
  rank~ : Int,
  device_ordinal~ : Int,
  memory_ceiling_bytes~ : Int64,
  weight_reserved_bytes~ : Int64,
  activation_workspace_reserved_bytes~ : Int64,
  kv_reserved_bytes~ : Int64,
  collective_reserved_bytes~ : Int64,
) -> TensorParallelRankRuntimeClaim raise WorkerWireError {
  if rank < 0 ||
    rank >= 16 ||
    device_ordinal < 0 ||
    device_ordinal > 1023 ||
    memory_ceiling_bytes <= 0L ||
    weight_reserved_bytes < 0L ||
    activation_workspace_reserved_bytes < 0L ||
    kv_reserved_bytes < 0L ||
    collective_reserved_bytes < 0L {
    raise InvalidBootstrapSource(Ceiling)
  }
  let mut reserved = weight_reserved_bytes
  if reserved > memory_ceiling_bytes - activation_workspace_reserved_bytes {
    raise InvalidBootstrapSource(Ceiling)
  }
  reserved += activation_workspace_reserved_bytes
  if reserved > memory_ceiling_bytes - kv_reserved_bytes {
    raise InvalidBootstrapSource(Ceiling)
  }
  reserved += kv_reserved_bytes
  if reserved > memory_ceiling_bytes - collective_reserved_bytes {
    raise InvalidBootstrapSource(Ceiling)
  }
  reserved += collective_reserved_bytes
  if reserved > memory_ceiling_bytes {
    raise InvalidBootstrapSource(Ceiling)
  }
  {
    rank,
    device_ordinal,
    memory_ceiling_bytes,
    weight_reserved_bytes,
    activation_workspace_reserved_bytes,
    kv_reserved_bytes,
    collective_reserved_bytes,
  }
}

///|
pub fn TensorParallelBootstrapPolicy::new(
  ranks : ArrayView[TensorParallelRankRuntimeClaim],
  minimum_collective_runtime_version_code~ : Int,
  maximum_collective_runtime_version_code~ : Int,
  collective_exchange_timeout_millis~ : Int,
) -> TensorParallelBootstrapPolicy raise WorkerWireError {
  if ranks.length() < 2 ||
    ranks.length() > 16 ||
    minimum_collective_runtime_version_code <= 0 ||
    maximum_collective_runtime_version_code <
    minimum_collective_runtime_version_code ||
    collective_exchange_timeout_millis <= 0 ||
    collective_exchange_timeout_millis > 3600000 {
    raise InvalidBootstrapSource(Ceiling)
  }
  let copied : Array[TensorParallelRankRuntimeClaim] = []
  for index, rank in ranks {
    if rank.rank != index {
      raise InvalidBootstrapSource(CanonicalEncoding)
    }
    copied.push(rank)
  }
  {
    ranks: ReadOnlyArray::from_array(copied),
    minimum_collective_runtime_version_code,
    maximum_collective_runtime_version_code,
    collective_exchange_timeout_millis,
  }
}

///|
pub fn TensorParallelBootstrapPolicy::rank_count(self : Self) -> Int {
  self.ranks.length()
}

///|
pub fn TensorParallelBootstrapPolicy::rank(
  self : Self,
  rank : Int,
) -> TensorParallelRankRuntimeClaim? {
  self.ranks.get(rank)
}

///|
pub fn TensorParallelBootstrapPolicy::minimum_collective_runtime_version_code(
  self : Self,
) -> Int {
  self.minimum_collective_runtime_version_code
}

///|
pub fn TensorParallelBootstrapPolicy::maximum_collective_runtime_version_code(
  self : Self,
) -> Int {
  self.maximum_collective_runtime_version_code
}

///|
/// Maximum monotonic duration of one complete rank-coordinated plan exchange.
/// The parent derives the absolute deadline from its owned clock at begin.
pub fn TensorParallelBootstrapPolicy::collective_exchange_timeout_millis(
  self : Self,
) -> Int {
  self.collective_exchange_timeout_millis
}

///|
pub fn TensorParallelRankRuntimeClaim::rank(self : Self) -> Int {
  self.rank
}

///|
pub fn TensorParallelRankRuntimeClaim::device_ordinal(self : Self) -> Int {
  self.device_ordinal
}

///|
pub fn TensorParallelRankRuntimeClaim::memory_ceiling_bytes(
  self : Self,
) -> Int64 {
  self.memory_ceiling_bytes
}

///|
pub fn TensorParallelRankRuntimeClaim::weight_reserved_bytes(
  self : Self,
) -> Int64 {
  self.weight_reserved_bytes
}

///|
pub fn TensorParallelRankRuntimeClaim::activation_workspace_reserved_bytes(
  self : Self,
) -> Int64 {
  self.activation_workspace_reserved_bytes
}

///|
pub fn TensorParallelRankRuntimeClaim::kv_reserved_bytes(self : Self) -> Int64 {
  self.kv_reserved_bytes
}

///|
pub fn TensorParallelRankRuntimeClaim::collective_reserved_bytes(
  self : Self,
) -> Int64 {
  self.collective_reserved_bytes
}
