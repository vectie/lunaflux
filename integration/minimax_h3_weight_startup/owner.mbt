///|
priv enum Owner {
  Packed(@device.Allocation)
  Encoder(@text.EncoderWeights)
  Visual(@vision.VisionWeights)
  Segmented(@segmented.SegmentedDeviceWeights, @media.WeightManifest)
  Cleanup(@segmented.FailedSegmentedDevicePreparation)
}

///|
priv struct Entry {
  owner : Owner
  bytes : Int64
  component : @weights.MiniMaxH3WeightComponent
  packing : Packing
}

///|
/// Startup-only aggregate owner. Successful and partially uploaded allocations
/// remain owned until reverse-order close. A failed prepare prohibits reuse.
pub struct WeightStartup {
  priv maximum : Int64
  priv entries : Array[Entry]
  priv completed : Array[@weights.MiniMaxH3WeightComponent]
  priv mut used : Int64
  priv mut failed : Bool
  priv mut closed : Bool
  priv mut leases : Int
  priv mut identity : @spec.ModelIdentity?
}

///|
pub fn WeightStartup::new(max_device_bytes~ : Int64) -> Self raise StartupError {
  guard max_device_bytes > 0L else { raise BudgetExceeded }
  {
    maximum: max_device_bytes,
    entries: [],
    completed: [],
    used: 0L,
    failed: false,
    closed: false,
    leases: 0,
    identity: None,
  }
}

///|
pub fn WeightStartup::reserved_bytes(self : Self) -> Int64 {
  self.used
}

///|
pub fn WeightStartup::count(self : Self) -> Int {
  self.entries.length()
}

///|
pub fn WeightStartup::component(
  self : Self,
  index : Int,
) -> @weights.MiniMaxH3WeightComponent raise StartupError {
  guard index >= 0 && index < self.entries.length() else { raise InvalidState }
  self.entries[index].component
}

///|
/// Lease spans all prepared downstream queues, not only the callback borrowing
/// addresses. Release it only after those queues have drained and closed.
pub struct WeightLease {
  priv owner : WeightStartup
  priv mut released : Bool
}

///|
pub fn WeightStartup::acquire(self : Self) -> WeightLease raise StartupError {
  guard !self.failed && !self.closed && !self.entries.is_empty() else {
    raise InvalidState
  }
  self.leases += 1
  { owner: self, released: false, }
}

///|
pub fn WeightLease::release(self : Self) -> Unit {
  if !self.released {
    self.owner.leases -= 1
    self.released = true
  }
}

///|
fn WeightLease::entry(self : Self, index : Int) -> Owner raise StartupError {
  guard !self.released &&
    !self.owner.failed &&
    !self.owner.closed &&
    index >= 0 &&
    index < self.owner.entries.length() else {
    raise InvalidState
  }
  self.owner.entries[index].owner
}

///|
pub fn[T] WeightLease::with_allocation(
  self : Self,
  index : Int,
  consume : (@device.Allocation) -> T raise,
) -> T raise {
  guard self.entry(index) is Packed(allocation) else { raise InvalidPacking }
  consume(allocation)
}

///|
pub fn[T] WeightLease::with_encoder(
  self : Self,
  index : Int,
  consume : (@text.EncoderWeights) -> T raise,
) -> T raise {
  guard self.entry(index) is Encoder(owner) else { raise InvalidPacking }
  consume(owner)
}

///|
pub fn[T] WeightLease::with_vision(
  self : Self,
  index : Int,
  consume : (@vision.VisionWeights) -> T raise,
) -> T raise {
  guard self.entry(index) is Visual(owner) else { raise InvalidPacking }
  consume(owner)
}

///|
pub fn[T] WeightLease::with_reference(
  self : Self,
  index : Int,
  consume : (@segmented.SegmentedDeviceWeights, @media.WeightManifest) -> T raise,
) -> T raise {
  guard self.entry(index) is Segmented(owner, manifest) else {
    raise InvalidPacking
  }
  consume(owner, manifest)
}

///|
pub fn WeightStartup::close(self : Self) -> Unit raise {
  guard self.leases == 0 else { raise InvalidState }
  self.failed = true
  while !self.entries.is_empty() {
    let i = self.entries.length() - 1
    let entry = self.entries[i]
    match entry.owner {
      Packed(a) => a.close()
      Encoder(a) => a.close()
      Visual(a) => a.close()
      Segmented(a, _) => a.close()
      Cleanup(a) => a.retry_close()
    }
    self.used -= entry.bytes
    self.entries.remove(i) |> ignore
  }
  self.completed.clear()
  self.closed = true
}
