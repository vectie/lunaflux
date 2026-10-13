///|
pub fn TensorCollectiveOwner::validate_group(
  self : Self,
  rank~ : Int,
  world_size~ : Int,
) -> Unit raise DeviceError {
  guard self.contract.rank == rank && self.contract.world_size == world_size else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
}

///|
pub fn TensorCollectiveOwner::validate_site_count(
  self : Self,
  count : Int,
) -> Unit raise DeviceError {
  guard count == self.contract.sequences.length() else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
}

///|
/// Startup shape check for a consumer's reduction boundary. This prevents a
/// job-by-channel partial tensor from accidentally binding a token-by-channel
/// collective. No device operation or request-path validation is performed.
pub fn TensorCollectiveOwner::validate_sum_site(
  self : Self,
  site_sequence~ : Int,
  operation_id~ : Int,
  maximum_rows~ : Int,
  width~ : Int,
) -> Unit raise DeviceError {
  let index = site_sequence - 1
  guard site_sequence > 0 &&
    index < self.contract.sequences.length() &&
    maximum_rows > 0 &&
    maximum_rows <= self.contract.max_query_tokens &&
    width > 0 &&
    self.contract.sequences[index] == site_sequence &&
    self.contract.operation_ids[index] == operation_id &&
    self.contract.kinds[index] == tensor_collective_kind_code(F32SumAllReduce) &&
    self.contract.full_widths[index] == width &&
    self.contract.local_widths[index] == width else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
}
