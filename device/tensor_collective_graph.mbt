///|
/// Seal a complete startup capture without publishing a runtime step. Operands
/// and communicator remain pinned until the borrowing graph is destroyed.
pub fn TensorCollectiveOwner::seal_graph(
  self : Self,
  previous_plan : UInt64,
) -> Unit raise DeviceError {
  self.require_live()
  let count = self.contract.sequences.length()
  guard self.next_site_index == count &&
    self.plan_sequence == previous_plan + 1UL &&
    self.next_global_sequence == count.to_uint64() + 1UL else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  self.communicator.seal_graph(previous_plan, count) catch {
    e => raise error_from_nccl_for(e, TensorCollectiveExecute)
  }
  self.plan_sequence = previous_plan
  self.next_global_sequence = 1UL
}

///|
pub fn TensorCollectiveOwner::begin_graph(
  self : Self,
  generation : UInt64,
  plan_sequence : UInt64,
  global_sequence : UInt64,
) -> Unit raise DeviceError {
  self.require_live()
  let count = self.contract.sequences.length()
  guard self.next_site_index == count &&
    generation == self.generation &&
    self.plan_sequence < 18446744073709551615UL &&
    plan_sequence == self.plan_sequence + 1UL &&
    global_sequence == self.next_global_sequence &&
    global_sequence <= 18446744073709551615UL - count.to_uint64() else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  self.communicator.begin_graph(generation, plan_sequence, global_sequence) catch {
    e => raise error_from_nccl_for(e, TensorCollectiveExecute)
  }
  self.pending_plan_sequence = plan_sequence
  self.pending_global_sequence = global_sequence + count.to_uint64() - 1UL
  self.pending_site_index = count - 1
  self.state = CollectiveOwnerInFlight
}

///|
/// Only after the borrowing graph's terminal device event.
pub fn TensorCollectiveOwner::complete_graph(
  self : Self,
) -> Bool raise DeviceError {
  guard self.state == CollectiveOwnerInFlight else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  let done = self.communicator.complete_graph() catch {
    e => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(e, TensorCollectiveExecute)
    }
  }
  if done {
    self.commit_submission()
  }
  done
}

///|
pub fn TensorCollectiveOwner::release_graph(
  self : Self,
) -> Unit raise DeviceError {
  self.require_live()
  self.communicator.release_graph() catch {
    e => raise error_from_nccl_for(e, TensorCollectiveExecute)
  }
}
