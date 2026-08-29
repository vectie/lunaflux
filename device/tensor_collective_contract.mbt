///|
pub fn TensorCollectiveContractDigest::from_sha256(
  value : StringView,
) -> TensorCollectiveContractDigest raise DeviceError {
  if value.length() != 64 {
    raise InvalidArgument(TensorCollectiveContractCreate)
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise InvalidArgument(TensorCollectiveContractCreate)
    }
  }
  { hex: value.to_owned() }
}

///|
pub fn TensorCollectiveContractDigest::as_hex(
  self : TensorCollectiveContractDigest,
) -> String {
  self.hex
}

///|
pub fn TensorCollectiveSite::new(
  sequence~ : Int,
  operation_id~ : Int,
  kind~ : TensorCollectiveKind,
  full_width~ : Int,
  local_width~ : Int,
) -> TensorCollectiveSite raise DeviceError {
  if sequence <= 0 || operation_id < 0 || full_width <= 0 || local_width <= 0 {
    raise InvalidArgument(TensorCollectiveContractCreate)
  }
  { sequence, operation_id, kind, full_width, local_width }
}

///|
fn tensor_collective_kind_code(kind : TensorCollectiveKind) -> Int {
  match kind {
    Bf16SumAllReduce => 1
    Bf16AllGather => 2
  }
}

///|
fn require_tensor_collective_widths(
  kind : TensorCollectiveKind,
  full_width : Int,
  local_width : Int,
  world_size : Int,
) -> Unit raise DeviceError {
  if full_width <= 0 || local_width <= 0 {
    raise InvalidArgument(TensorCollectiveContractCreate)
  }
  match kind {
    Bf16SumAllReduce =>
      if full_width != local_width {
        raise InvalidArgument(TensorCollectiveContractCreate)
      }
    Bf16AllGather =>
      if full_width % world_size != 0 || full_width / world_size != local_width {
        raise InvalidArgument(TensorCollectiveContractCreate)
      }
  }
}

///|
/// Copies one authenticated scalar site table into fixed device-owned storage.
pub fn TensorCollectiveContract::new(
  digest~ : TensorCollectiveContractDigest,
  rank~ : Int,
  world_size~ : Int,
  max_query_tokens~ : Int,
  sites~ : ArrayView[TensorCollectiveSite],
) -> Self raise DeviceError {
  if world_size < 2 ||
    world_size > 16 ||
    rank < 0 ||
    rank >= world_size ||
    max_query_tokens <= 0 ||
    sites.length() == 0 {
    raise InvalidArgument(TensorCollectiveContractCreate)
  }
  let count = sites.length()
  let sequences = FixedArray::make(count, 0)
  let operation_ids = FixedArray::make(count, 0)
  let kinds = FixedArray::make(count, 0)
  let full_widths = FixedArray::make(count, 0)
  let local_widths = FixedArray::make(count, 0)
  let mut last_operation_id = -1
  for index in 0..<count {
    let site = sites[index]
    require_tensor_collective_widths(
      site.kind,
      site.full_width,
      site.local_width,
      world_size,
    )
    try {
      ignore(
        checked_tensor_collective_elements(
          max_query_tokens.to_int64(),
          site.full_width.to_int64(),
        ),
      )
      ignore(
        checked_tensor_collective_elements(
          max_query_tokens.to_int64(),
          site.local_width.to_int64(),
        ),
      )
    } catch {
      _ => raise InvalidArgument(TensorCollectiveContractCreate)
    }
    if site.sequence != index + 1 || site.operation_id <= last_operation_id {
      raise InvalidArgument(TensorCollectiveContractCreate)
    }
    sequences[index] = site.sequence
    operation_ids[index] = site.operation_id
    kinds[index] = tensor_collective_kind_code(site.kind)
    full_widths[index] = site.full_width
    local_widths[index] = site.local_width
    last_operation_id = site.operation_id
  }
  {
    digest,
    rank,
    world_size,
    max_query_tokens,
    sequences,
    operation_ids,
    kinds,
    full_widths,
    local_widths,
  }
}

///|
fn checked_tensor_collective_elements(
  query_tokens : Int64,
  width : Int64,
) -> Int64 raise DeviceError {
  if query_tokens <= 0L ||
    width <= 0L ||
    query_tokens > 9223372036854775807L / width {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  query_tokens * width
}

///|
pub fn TensorCollectiveContract::digest(
  self : TensorCollectiveContract,
) -> TensorCollectiveContractDigest {
  self.digest
}

///|
pub fn TensorCollectiveContract::rank(self : TensorCollectiveContract) -> Int {
  self.rank
}

///|
pub fn TensorCollectiveContract::world_size(
  self : TensorCollectiveContract,
) -> Int {
  self.world_size
}

///|
/// Returns the startup-authenticated maximum live query-token envelope.
pub fn TensorCollectiveContract::max_query_tokens(
  self : TensorCollectiveContract,
) -> Int {
  self.max_query_tokens
}

///|
pub fn TensorCollectiveContract::site_count(
  self : TensorCollectiveContract,
) -> Int {
  self.sequences.length()
}
