///|
fn validate_token_ids(
  token_bytes : Bytes,
  vocabulary_size : Int,
) -> Unit raise DeviceExecutorError {
  let token_count = token_bytes.length() / 4
  for index in 0..<token_count {
    let offset = index * 4
    let high = token_bytes[offset + 3].to_int()
    if (high & 0x80) != 0 {
      raise InvalidTokenId(index~)
    }
    let token_id = token_bytes[offset].to_int() |
      (token_bytes[offset + 1].to_int() << 8) |
      (token_bytes[offset + 2].to_int() << 16) |
      (high << 24)
    if token_id >= vocabulary_size {
      raise InvalidTokenId(index~)
    }
  }
}

///|
/// Dispatches the exact prepared full-sequence profile.
///
/// `token_bytes` must contain little-endian signed Int32 token IDs. Their exact
/// count and vocabulary range are checked before the device copy. All launch
/// arguments and vendor descriptors were prepared earlier; this path creates
/// no metadata arrays, maps, strings, kernel plans, or device allocations.
pub fn PreparedDeviceExecutor::dispatch(
  self : PreparedDeviceExecutor,
  token_bytes : Bytes,
) -> Unit raise DeviceExecutorError {
  require_ready(self.lifecycle)
  let actual = Int64::from_int(token_bytes.length())
  if actual != self.token_bytes {
    raise InvalidTokenBytes(expected=self.token_bytes, actual~)
  }
  validate_token_ids(token_bytes, self.vocabulary_size)
  guard self.resources.stream is Some(stream) && stream.open else {
    raise Closed
  }
  guard self.resources.token_staging is Some(token_staging) &&
    token_staging.open else {
    raise Closed
  }
  guard self.resources.arena is Some(arena) && arena.open else { raise Closed }
  self.output_valid = false
  token_staging.resource.copy_from_host(token_bytes, 0L, 0L, self.token_bytes) catch {
    error => raise DeviceFailed(TokenTransfer, error)
  }
  for step in self.steps {
    match step {
      Aot(aot) => {
        guard self.resources.functions.get(aot.function_index) is Some(function) &&
          function.open else {
          raise Closed
        }
        function.resource.launch_synchronous(
          aot.dimensions,
          aot.arguments,
          stream.resource,
        ) catch {
          error => raise DeviceFailed(AotLaunch, error)
        }
      }
      Vendor(vendor) => {
        guard self.resources.vendor_plans.get(vendor.plan_index) is Some(plan) &&
          plan.open else {
          raise Closed
        }
        plan.resource.run(
          arena.resource.allocation(),
          vendor.left_offset,
          self.weight_allocation,
          vendor.weight_offset,
          arena.resource.allocation(),
          vendor.output_offset,
          arena.resource.allocation(),
          vendor.workspace_offset,
          stream.resource,
        ) catch {
          error => raise DeviceFailed(VendorGemmRun, error)
        }
      }
    }
  }
  self.output_valid = true
}

///|
/// Copies one precomputed terminal row into a fresh bounded host buffer.
/// This is a Phase-1 correctness API, not a steady-state sampling path.
pub fn PreparedDeviceExecutor::copy_terminal_row(
  self : PreparedDeviceExecutor,
  batch_index : Int,
) -> Bytes raise DeviceExecutorError {
  require_ready(self.lifecycle)
  if !self.output_valid {
    raise OutputUnavailable
  }
  guard self.terminal_output.selected_rows().get(batch_index) is Some(row) &&
    row.batch_index() == batch_index else {
    raise InvalidOutputRow(index=batch_index)
  }
  guard self.resources.arena is Some(arena) && arena.open else { raise Closed }
  arena.resource
  .allocation()
  .copy_to_host(row.offset_bytes(), row.byte_length()) catch {
    error => raise DeviceFailed(OutputTransfer, error)
  }
}
