///|
/// Fixed capacities admitted before any storage is allocated.
#valtype
pub struct Limits {
  slots : Int
  transfers : Int
  bytes_per_slot : Int64
  host_bytes : Int64
}

///|
pub(all) suberror ResidencyError {
  InvalidLimits
}

///|
/// The capacity tier requires physically separate host and device memory.
pub fn Limits::new(
  slots~ : Int,
  transfers~ : Int,
  bytes_per_slot~ : Int64,
  host_budget~ : Int64,
  separate_host_memory~ : Bool,
) -> Limits raise ResidencyError {
  if !separate_host_memory ||
    slots <= 0 ||
    transfers <= 0 ||
    transfers > slots ||
    bytes_per_slot <= 0L ||
    host_budget <= 0L ||
    bytes_per_slot > host_budget / slots.to_int64() {
    raise InvalidLimits
  }
  {
    slots,
    transfers,
    bytes_per_slot,
    host_bytes: bytes_per_slot * slots.to_int64(),
  }
}

///|
pub fn Limits::host_bytes(self : Limits) -> Int64 {
  self.host_bytes
}

///|
pub fn Limits::slots(self : Limits) -> Int {
  self.slots
}

///|
pub fn Limits::transfers(self : Limits) -> Int {
  self.transfers
}

///|
pub fn Limits::bytes_per_slot(self : Limits) -> Int64 {
  self.bytes_per_slot
}

///|
#valtype
priv enum Phase {
  Free
  Spilling
  Host
  Restoring
  Quarantined
  Retired
}

///|
#valtype
priv struct State {
  phase : Phase
  generation : Int
  epoch : Int
  cancelled : Bool
}

///|
#valtype
priv enum Event {
  BeginSpill
  BeginRestore
  Cancel
  Complete
  Fail
  Drained
  Evict
}

///|
#valtype
pub(all) enum Effect {
  Rejected
  CopyToHost
  CopyToDevice
  RetainUntilCompletion
  HostReady
  DeviceReady
  RestoreCancelled
  DrainRequired
  Released
}

///|
#valtype
priv struct Transition {
  state : State
  effect : Effect
}

///|
/// Opaque owner-scoped cache identity. It grants no device/page authority.
#valtype
pub struct Entry {
  priv owner : Pool
  priv slot : Int
  priv generation : Int
}

///|
/// An exact transfer attempt; old and duplicate completions are rejected.
#valtype
pub struct Transfer {
  priv entry : Entry
  priv epoch : Int
}

///|
#valtype
pub(all) enum Begin {
  Started(Transfer)
  Backpressured
  InvalidEntry
  Closed
}

///|
/// Mutable storage is private; transitions and admission decisions are pure.
pub struct Pool {
  priv limits : Limits
  priv states : FixedArray[State]
  priv free : FixedArray[Int]
  priv mut free_count : Int
  priv mut inflight : Int
  priv mut closed : Bool
}

///|
pub fn Transfer::entry(self : Transfer) -> Entry {
  self.entry
}

///|
pub fn Transfer::slot(self : Transfer) -> Int {
  self.entry.slot
}

///|
pub fn Entry::slot(self : Entry) -> Int {
  self.slot
}

///|
pub fn Entry::generation(self : Entry) -> Int {
  self.generation
}

///|
pub fn Transfer::same_attempt(self : Transfer, other : Transfer) -> Bool {
  physical_equal(self.entry.owner, other.entry.owner) &&
  self.entry.slot == other.entry.slot &&
  self.entry.generation == other.entry.generation &&
  self.epoch == other.epoch
}
