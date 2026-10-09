///|
pub(all) suberror StreamingError {
  InvalidCapability
  InvalidFrame
  InvalidTransition
}

///|
/// Startup-only, immutable capacity and measured-policy contract. Host memory
/// must be a separate physical tier. Costs describe one complete logical page.
pub struct Capability {
  slots : Int
  host_bytes : Int64
  page_bytes : Int64
  page_count : Int
  reserve_pages : Int
  restore_ns : Int64
  recompute_ns : Int64
  transfer_timeout_millis : Int
  workspace_budget : Int64
  shapes : ReadOnlyArray[Shape]
} derive(Eq, Debug)

///|
pub extend Capability with Eq::{equal, not_equal}

///|
pub extend Capability with Debug::{to_repr}

///|
pub fn Capability::new(
  slots~ : Int,
  host_bytes~ : Int64,
  page_bytes~ : Int64,
  page_count~ : Int,
  reserve_pages~ : Int,
  restore_ns~ : Int64,
  recompute_ns~ : Int64,
  separate_host_memory~ : Bool,
  transfer_timeout_millis? : Int = 10000,
  workspace_budget? : Int64 = 0L,
  shapes? : ArrayView[Shape] = [],
) -> Capability raise StreamingError {
  if !separate_host_memory ||
    slots <= 0 ||
    slots > 65536 ||
    page_bytes <= 0L ||
    host_bytes <= 0L ||
    page_bytes > host_bytes / slots.to_int64() ||
    page_count <= 0 ||
    page_count > 1048576 ||
    reserve_pages < 1 ||
    reserve_pages > page_count ||
    restore_ns <= 0L ||
    recompute_ns <= 0L ||
    transfer_timeout_millis <= 0 ||
    transfer_timeout_millis > 600000 ||
    workspace_budget < 0L ||
    shapes.length() > 32 ||
    (shapes.length() == 0 && workspace_budget != 0L) {
    raise InvalidCapability
  }
  for i in 0..<shapes.length() {
    for j in 0..<i {
      if shapes[i].tokens == shapes[j].tokens {
        raise InvalidCapability
      }
    }
  }
  {
    slots,
    host_bytes,
    page_bytes,
    page_count,
    reserve_pages,
    restore_ns,
    recompute_ns,
    transfer_timeout_millis,
    workspace_budget,
    shapes: ReadOnlyArray::from_array(shapes.to_owned()),
  }
}

///|
pub fn Capability::slots(self : Capability) -> Int {
  self.slots
}

///|
pub fn Capability::host_bytes(self : Capability) -> Int64 {
  self.host_bytes
}

///|
pub fn Capability::page_bytes(self : Capability) -> Int64 {
  self.page_bytes
}

///|
pub fn Capability::page_count(self : Capability) -> Int {
  self.page_count
}

///|
pub fn Capability::reserve_pages(self : Capability) -> Int {
  self.reserve_pages
}

///|
pub fn Capability::restore_ns(self : Capability) -> Int64 {
  self.restore_ns
}

///|
pub fn Capability::recompute_ns(self : Capability) -> Int64 {
  self.recompute_ns
}

///|
pub fn Capability::transfer_timeout_millis(self : Capability) -> Int {
  self.transfer_timeout_millis
}

///|
pub fn Capability::workspace_budget(self : Capability) -> Int64 {
  self.workspace_budget
}

///|
pub fn Capability::shape_count(self : Capability) -> Int {
  self.shapes.length()
}

///|
pub fn Capability::shape_at(self : Capability, index : Int) -> Shape {
  self.shapes[index]
}

///|
#valtype
pub struct Shape {
  tokens : Int
  workspace_bytes : Int64
} derive(Eq, Debug)

///|
pub extend Shape with Eq::{equal, not_equal}

///|
pub extend Shape with Debug::{to_repr}

///|
pub fn Shape::new(
  tokens~ : Int,
  workspace_bytes~ : Int64,
) -> Shape raise StreamingError {
  if tokens <= 0 || tokens > 1048576 || workspace_bytes < 0L {
    raise InvalidCapability
  }
  { tokens, workspace_bytes, }
}

///|
pub fn Shape::tokens(self : Shape) -> Int {
  self.tokens
}

///|
pub fn Shape::workspace_bytes(self : Shape) -> Int64 {
  self.workspace_bytes
}

///|
/// No owner references or pointers cross the process boundary.
#valtype
pub(all) struct HostPage {
  slot : Int
  generation : Int
}

///|
#valtype
pub(all) enum Opcode {
  Spill
  Restore
  Poll
  Cancel
  Evict
}

///|
#valtype
pub(all) struct Command {
  worker : UInt64
  sequence : UInt64
  opcode : Opcode
  page : Int
  page_generation : Int
  host : HostPage
}

///|
#valtype
pub(all) enum Status {
  Pending
  HostReady
  DeviceReady
  Cancelled
  Evicted
  Rejected
}

///|
#valtype
pub(all) struct Reply {
  command : Command
  status : Status
  host : HostPage
}

///|
/// Allocation-free optional effect for the scheduler's steady path.
#valtype
pub(all) enum CommandOption {
  None
  Some(Command)
}

///|
/// Requires `Some`, matching the usual Option unwrap contract.
pub fn CommandOption::unwrap(self : CommandOption) -> Command {
  match self {
    Some(command) => command
    None => abort("no streaming command")
  }
}

///|
#valtype
pub(all) enum ReplyOption {
  None
  Some(Reply)
}

///|
pub fn HostPage::same(self : HostPage, other : HostPage) -> Bool {
  self.slot == other.slot && self.generation == other.generation
}

///|
pub fn Command::same_transaction(self : Command, other : Command) -> Bool {
  self.worker == other.worker &&
  self.sequence == other.sequence &&
  self.page == other.page &&
  self.page_generation == other.page_generation &&
  self.host.same(other.host)
}

///|
pub fn Command::same_command(self : Command, other : Command) -> Bool {
  if !self.same_transaction(other) {
    return false
  }
  match (self.opcode, other.opcode) {
    (Spill, Spill)
    | (Restore, Restore)
    | (Poll, Poll)
    | (Cancel, Cancel)
    | (Evict, Evict) => true
    _ => false
  }
}

///|
pub fn Command::valid(
  self : Command,
  capability : Capability,
  worker : UInt64,
) -> Bool {
  if self.worker != worker ||
    worker == 0UL ||
    self.sequence == 0UL ||
    self.page < 0 ||
    self.page >= capability.page_count ||
    self.page_generation <= 0 {
    return false
  }
  match self.opcode {
    Spill => self.host.slot == -1 && self.host.generation == 0
    Restore | Evict =>
      self.host.slot >= 0 &&
      self.host.slot < capability.slots &&
      self.host.generation > 0
    Poll | Cancel =>
      (self.host.slot == -1 && self.host.generation == 0) ||
      (
        self.host.slot >= 0 &&
        self.host.slot < capability.slots &&
        self.host.generation > 0
      )
  }
}
