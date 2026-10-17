///|
/// Fixed output-domain transport, independent of the decoder input row count.
/// This is used for compact rank-major candidate records, not a numeric sum.
pub fn ReductionBoundary::fixed_gather(
  before_kernel~ : Int,
  local_width~ : Int,
  world~ : Int,
  operation_id~ : Int,
  send~ : @device.Allocation,
  receive~ : @device.Allocation,
) -> Self raise @device.DeviceError {
  guard local_width > 0 && world > 1 && local_width <= 2147483647 / world else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  {
    before_kernel,
    width: local_width * world,
    operation_id,
    buffer: send,
    receive,
    local_width,
    fixed_rows: Some(1),
    kind: F32AllGather,
  }
}

///|
fn ReductionBoundary::row_count(self : Self, live : Int) -> Int {
  self.fixed_rows.unwrap_or(live)
}
