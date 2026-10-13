///|
/// Startup-owned storage for outstanding same-stream collective operands.
pub fn TensorCollectiveOwner::prepare_deferred(
  self : Self,
) -> Unit raise DeviceError {
  self.require_live()
  self.communicator.prepare_deferred(self.contract.sequences.length()) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveExecute)
  }
}

///|
fn TensorCollectiveOwner::commit_submission(self : Self) -> Unit {
  self.plan_sequence = self.pending_plan_sequence
  self.next_global_sequence = self.pending_global_sequence + 1UL
  self.next_site_index = self.pending_site_index + 1
  self.pending_plan_sequence = 0UL
  self.pending_global_sequence = 0UL
  self.pending_site_index = 0
  self.state = CollectiveOwnerLive
}

///|
/// Advances submission only. Native leases remain retained until poll_deferred
/// observes completion; close remains Busy and abort remains consuming.
pub fn TensorCollectiveOwner::defer_submitted(
  self : Self,
) -> Bool raise DeviceError {
  guard self.state == CollectiveOwnerInFlight else {
    raise InvalidArgument(TensorCollectiveExecute)
  }
  let submitted = self.communicator.defer_submitted() catch {
    error => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(error, TensorCollectiveExecute)
    }
  }
  if submitted {
    self.commit_submission()
  }
  submitted
}

///|
pub fn TensorCollectiveOwner::poll_deferred(
  self : Self,
) -> Bool raise DeviceError {
  self.require_live()
  self.communicator.poll_deferred() catch {
    error => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(error, TensorCollectiveExecute)
    }
  }
}
