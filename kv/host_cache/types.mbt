///|
#valtype
priv enum Lane {
  Idle
  Active(@residency.Transfer)
  Fault(@residency.Transfer)
}

///|
/// Worker-local owner. Caller retains device-page reservations until an
/// operation completes or explicit drain succeeds. This stores bytes; logical
/// prefix identity and active-reference admission belong to the prefix owner.
pub struct Cache {
  priv plan : Plan
  priv residency : @residency.Pool
  priv transfers : @device.TransferPool
  priv lanes : FixedArray[Lane]
  priv mut poisoned : Bool
  priv mut closed : Bool
}

///|
#valtype
pub struct Operation {
  priv owner : Cache
  priv lane : Int
  priv transfer : @residency.Transfer
}

///|
#valtype
pub(all) enum Start {
  Started(Operation)
  Backpressured
  Rejected
}

///|
#valtype
pub(all) enum Progress {
  Pending
  HostReady
  DeviceReady
  Cancelled
  Stale
}

///|
pub fn Operation::entry(self : Operation) -> @residency.Entry {
  self.transfer.entry()
}

///|
pub fn Cache::plan(self : Cache) -> Plan {
  self.plan
}

///|
pub fn Cache::is_idle(self : Cache) -> Bool {
  !self.closed && !self.poisoned && self.residency.inflight() == 0
}

///|
pub fn Cache::is_prepared(self : Cache) -> Bool {
  !self.closed && !self.poisoned && self.transfers.is_prepared()
}
