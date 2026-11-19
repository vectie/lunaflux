///|
priv enum LoadState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
pub struct Loaded {
  priv plan : WeightPlan
  priv allocations : Array[@device.Allocation]
  priv mut weights : Weights?
  priv mut state : LoadState
}

///|
pub fn Loaded::new(plan : WeightPlan) -> Self {
  { plan, allocations: [], weights: None, state: Fresh, }
}

///|
/// Admission precedes allocation. Close this owner after any partial failure.
pub fn Loaded::prepare(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_device_bytes >= self.plan.bytes &&
    scratch.length() > 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  for entry in self.plan.entries {
    self.allocations.push(context.allocate(entry.bytes))
  }
  let primary : Ref[@device.DeviceError?] = Ref::new(None)
  self.plan.transfer(root, scratch~, limits~, consume=(
    ordinal,
    offset,
    bytes,
    count,
  ) => {
    try
      context.copy_from_fixed_host(
        self.allocations[ordinal],
        bytes,
        0L,
        offset,
        count.to_int64(),
      )
    catch {
      error => {
        primary.val = Some(error)
        false
      }
    } noraise {
      _ => true
    }
  }) catch {
    error =>
      match primary.val {
        Some(native) => raise native
        None => raise error
      }
  }
  let regions : Map[String, WeightRegion] = Map([])
  for i in 0..<self.plan.entries.length() {
    let entry = self.plan.entries[i]
    regions[entry.name] = {
      allocation: self.allocations[i],
      bytes: entry.bytes,
    }
  }
  self.weights = Some({ regions, })
  self.state = Prepared
}

///|
pub fn Loaded::weights(self : Self) -> Weights raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.weights.unwrap()
}

///|
pub fn Loaded::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.weights = None
  while self.allocations.last() is Some(allocation) {
    allocation.close()
    ignore(self.allocations.pop())
  }
  self.state = Closed
}
