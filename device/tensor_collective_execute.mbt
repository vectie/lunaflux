///|
fn TensorCollectiveOwner::fail_authentication(
  self : TensorCollectiveOwner,
) -> Unit raise DeviceError {
  self.state = CollectiveOwnerFailed
  self.communicator.invalidate(generation=self.generation) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveInvalidate)
  }
  raise InvalidArgument(TensorCollectiveExecute)
}

///|
fn TensorCollectiveOwner::require_live(
  self : TensorCollectiveOwner,
) -> Unit raise DeviceError {
  match self.state {
    CollectiveOwnerLive => ()
    CollectiveOwnerStarting => raise Busy(TensorCollectiveExecute)
    CollectiveOwnerInFlight => raise Busy(TensorCollectiveExecute)
    CollectiveOwnerFailed => raise DriverFailure(TensorCollectiveExecute)
    CollectiveOwnerClosed => raise Closed(TensorCollectiveExecute)
  }
}

///|
fn require_tensor_collective_query_tokens(
  contract : TensorCollectiveContract,
  live_query_tokens : Int,
) -> Unit raise DeviceError {
  if live_query_tokens <= 0 || live_query_tokens > contract.max_query_tokens {
    raise InvalidArgument(TensorCollectiveExecute)
  }
}

///|
fn authenticate_tensor_collective_claim(
  contract : TensorCollectiveContract,
  generation : UInt64,
  current_plan_sequence : UInt64,
  next_global_sequence : UInt64,
  next_site_index : Int,
  claimed_generation : UInt64,
  claimed_plan_sequence : UInt64,
  claimed_global_sequence : UInt64,
  claimed_site_sequence : Int,
  claimed_operation_id : Int,
  claimed_kind : TensorCollectiveKind,
) -> Int raise DeviceError {
  let count = contract.sequences.length()
  let mut index = next_site_index
  if claimed_plan_sequence != current_plan_sequence {
    if next_site_index != count ||
      current_plan_sequence == 18446744073709551615UL ||
      claimed_plan_sequence != current_plan_sequence + 1UL {
      raise InvalidArgument(TensorCollectiveExecute)
    }
    index = 0
  } else if index >= count {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  if claimed_generation != generation ||
    claimed_global_sequence != next_global_sequence ||
    claimed_global_sequence == 18446744073709551615UL ||
    claimed_site_sequence != index + 1 ||
    claimed_site_sequence != contract.sequences[index] ||
    claimed_operation_id != contract.operation_ids[index] ||
    tensor_collective_kind_code(claimed_kind) != contract.kinds[index] {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  index
}

///|
fn nccl_collective_kind(
  kind : TensorCollectiveKind,
) -> @nccl.NcclCollectiveOperation {
  match kind {
    Bf16SumAllReduce => AllReduce
    Bf16AllGather => AllGather
  }
}

///|
/// Authenticates the immutable contract and every scalar launch identity, then
/// enters one native authenticate/acquire/enqueue boundary. Poll is the only
/// sequence-commit point. This owner is thread-confined; submit, poll, and
/// terminal methods must not race.
pub fn TensorCollectiveOwner::submit(
  self : TensorCollectiveOwner,
  generation~ : UInt64,
  plan_sequence~ : UInt64,
  global_sequence~ : UInt64,
  site_sequence~ : Int,
  operation_id~ : Int,
  kind~ : TensorCollectiveKind,
  live_query_tokens~ : Int,
  send~ : Allocation,
  send_offset~ : Int64,
  receive~ : Allocation,
  receive_offset~ : Int64,
  stream~ : Stream,
) -> Unit raise DeviceError {
  @cuda.link_collective_interop()
  self.require_live()
  require_tensor_collective_query_tokens(self.contract, live_query_tokens) catch {
    _ => self.fail_authentication()
  }
  let index = authenticate_tensor_collective_claim(
    self.contract,
    self.generation,
    self.plan_sequence,
    self.next_global_sequence,
    self.next_site_index,
    generation,
    plan_sequence,
    global_sequence,
    site_sequence,
    operation_id,
    kind,
  ) catch {
    _ => {
      self.fail_authentication()
      0
    }
  }
  let send_width = match kind {
    Bf16SumAllReduce => self.contract.full_widths[index]
    Bf16AllGather => self.contract.local_widths[index]
  }
  let send_elements = checked_tensor_collective_elements(
    live_query_tokens.to_int64(),
    send_width.to_int64(),
  ) catch {
    _ => {
      self.fail_authentication()
      0L
    }
  }
  let receive_elements = checked_tensor_collective_elements(
    live_query_tokens.to_int64(),
    self.contract.full_widths[index].to_int64(),
  ) catch {
    _ => {
      self.fail_authentication()
      0L
    }
  }
  self.communicator.submit_bf16(
    generation=self.generation,
    plan_sequence~,
    collective_sequence=global_sequence,
    operation_id~,
    collective_kind=nccl_collective_kind(kind),
    context=self.context.raw.collective_token(),
    send=send.raw.collective_token(),
    send_offset~,
    send_elements~,
    receive=receive.raw.collective_token(),
    receive_offset~,
    receive_elements~,
    stream=stream.raw.collective_token(),
  ) catch {
    error => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(error, TensorCollectiveExecute)
    }
  }
  self.pending_plan_sequence = plan_sequence
  self.pending_global_sequence = global_sequence
  self.pending_site_index = index
  self.state = CollectiveOwnerInFlight
}

///|
/// Polls one submitted collective. False is pending; true is the sole point
/// where the owner publishes scalar sequence progress.
pub fn TensorCollectiveOwner::poll(
  self : TensorCollectiveOwner,
) -> Bool raise DeviceError {
  @cuda.link_collective_interop()
  match self.state {
    CollectiveOwnerInFlight => ()
    CollectiveOwnerStarting => raise Busy(TensorCollectiveExecute)
    CollectiveOwnerLive => raise InvalidArgument(TensorCollectiveExecute)
    CollectiveOwnerFailed => raise DriverFailure(TensorCollectiveExecute)
    CollectiveOwnerClosed => raise Closed(TensorCollectiveExecute)
  }
  let completed = self.communicator.poll_collective() catch {
    error => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(error, TensorCollectiveExecute)
    }
  }
  if completed {
    self.plan_sequence = self.pending_plan_sequence
    self.next_global_sequence = self.pending_global_sequence + 1UL
    self.next_site_index = self.pending_site_index + 1
    self.pending_plan_sequence = 0UL
    self.pending_global_sequence = 0UL
    self.pending_site_index = 0
    self.state = CollectiveOwnerLive
  }
  completed
}

///|
/// Latches this exact generation failed. Abort is then the sole cleanup path.
pub fn TensorCollectiveOwner::invalidate(
  self : TensorCollectiveOwner,
) -> Unit raise DeviceError {
  @cuda.link_collective_interop()
  match self.state {
    CollectiveOwnerStarting | CollectiveOwnerLive | CollectiveOwnerInFlight =>
      ()
    CollectiveOwnerFailed => return
    CollectiveOwnerClosed => raise Closed(TensorCollectiveInvalidate)
  }
  self.state = CollectiveOwnerFailed
  self.communicator.invalidate(generation=self.generation) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveInvalidate)
  }
}

///|
/// Collectively consumes a healthy communicator. The owner remains terminal
/// even when the consuming native destroy reports failure.
pub fn TensorCollectiveOwner::close(
  self : TensorCollectiveOwner,
) -> Unit raise DeviceError {
  @cuda.link_collective_interop()
  match self.state {
    CollectiveOwnerClosed => return
    CollectiveOwnerFailed => raise DriverFailure(TensorCollectiveClose)
    CollectiveOwnerStarting | CollectiveOwnerInFlight =>
      raise Busy(TensorCollectiveClose)
    CollectiveOwnerLive => ()
  }
  self.state = CollectiveOwnerClosed
  self.communicator.close() catch {
    error => raise error_from_nccl_for(error, TensorCollectiveClose)
  }
}

///|
/// Consumes a failed or draining communicator exactly once.
pub fn TensorCollectiveOwner::abort(
  self : TensorCollectiveOwner,
) -> Unit raise DeviceError {
  @cuda.link_collective_interop()
  if self.state == CollectiveOwnerClosed {
    return
  }
  self.state = CollectiveOwnerClosed
  self.communicator.abort() catch {
    error => raise error_from_nccl_for(error, TensorCollectiveAbort)
  }
}
