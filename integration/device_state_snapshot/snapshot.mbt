///|
priv enum Phase {
  Ready
  Saving
  Tentative
  Restoring
  Failed
  Closed
}

///|
/// One reusable state transaction, prepared before token execution. The owner
/// must join model completion before rollback/commit; all state mutation must
/// use the supplied stream or an explicitly ordered dependency.
pub struct DeviceStateSnapshot {
  priv plan : @plan.StateSnapshotPlan
  priv backup : @device.Allocation
  priv copy : @device.DeviceCopyBatch
  priv mut copy_closed : Bool
  priv mut phase : Phase
}

///|
pub fn prepare_device_state_snapshot(
  context : @device.Context,
  stream : @device.Stream,
  state : ArrayView[@device.Allocation],
  maximum_backup_bytes~ : Int64,
) -> DeviceStateSnapshot raise {
  let plan = @plan.plan_state_snapshot(
    state.map(allocation => allocation.byte_count()),
    maximum_backup_bytes~,
  )
  let backup = context.allocate(plan.backup_bytes())
  let copy = try {
    let regions = state.mapi((i, allocation) => {
      @device.DeviceCopyRegion::new(
        allocation,
        backup,
        destination_offset=plan.offsets()[i],
        byte_count=plan.sizes()[i],
      )
    })
    context.prepare_device_copy(stream, regions)
  } catch {
    error => {
      backup.close()
      raise error
    }
  }
  { plan, backup, copy, copy_closed: false, phase: Ready, }
}

///|
pub fn DeviceStateSnapshot::backup_bytes(self : Self) -> Int64 {
  self.plan.backup_bytes()
}

///|
pub fn DeviceStateSnapshot::begin(self : Self) -> Unit raise {
  guard self.phase is Ready else {
    raise @device.DeviceError::InvalidArgument(DeviceCopySubmit)
  }
  self.phase = Failed
  self.copy.submit()
  self.phase = Saving
}

///|
pub fn DeviceStateSnapshot::poll_completion(self : Self) -> Bool raise {
  guard self.phase is (Saving | Restoring) else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyPoll)
  }
  let phase = self.phase
  let ready = self.copy.poll_completion() catch {
    error => {
      self.phase = Failed
      raise error
    }
  }
  if ready {
    self.phase = if phase is Saving { Tentative } else { Ready }
  }
  ready
}

///|
/// Wait is for teardown/reference runners; serving can poll the one event.
pub fn DeviceStateSnapshot::wait_completion(self : Self) -> Unit raise {
  guard self.phase is (Saving | Restoring) else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyWait)
  }
  let phase = self.phase
  self.phase = Failed
  self.copy.wait_completion()
  self.phase = if phase is Saving { Tentative } else { Ready }
}

///|
/// Keeps the verified mutations. Call only after the model transaction retires.
pub fn DeviceStateSnapshot::commit(self : Self) -> Unit raise {
  guard self.phase is Tentative else {
    raise @device.DeviceError::InvalidArgument(DeviceCopySubmit)
  }
  self.phase = Ready
}

///|
/// Restores payload and metadata together. Rejected suffixes are not removed by
/// merely changing a logical frontier: overwritten rings/pool state also undo.
pub fn DeviceStateSnapshot::rollback(self : Self) -> Unit raise {
  guard self.phase is Tentative else {
    raise @device.DeviceError::InvalidArgument(DeviceCopySubmit)
  }
  self.phase = Failed
  self.copy.submit(reverse=true)
  self.phase = Restoring
}

///|
/// Terminal cancellation cleanup after the caller has drained all model users.
/// A tentative backup is discarded, not promoted or restored into a live
/// request. Copies retire before leases disappear; no publication is possible
/// because the caller is closing the corresponding rank.
pub fn DeviceStateSnapshot::close_after_retirement(self : Self) -> Unit raise {
  if self.phase is Closed {
    return
  }
  self.phase = Failed
  self.close()
}

///|
/// Pending transfers are drained. An unresolved tentative transaction must be
/// explicitly committed or rolled back before a healthy owner is released.
pub fn DeviceStateSnapshot::close(self : Self) -> Unit raise {
  if self.phase is Closed {
    return
  }
  guard self.phase is (Ready | Failed) else {
    raise @device.DeviceError::Busy(DeviceCopyClose)
  }
  self.phase = Failed
  if !self.copy_closed {
    self.copy.wait_completion()
    self.copy.close()
    self.copy_closed = true
  }
  self.backup.close()
  self.phase = Closed
}
