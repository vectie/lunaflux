///|
pub fn Arena::byte_capacity(self : Arena) -> Int64 {
  self.bytes
}

///|
pub fn Arena::initially_initialized(self : Arena) -> Bool {
  self.initialized
}

///|
pub fn Buffer::arena(self : Buffer) -> Int {
  self.arena
}

///|
pub fn Buffer::offset_bytes(self : Buffer) -> Int64 {
  self.offset
}

///|
pub fn Buffer::byte_count(self : Buffer) -> Int64 {
  self.bytes
}

///|
pub fn Access::buffer(self : Access) -> Int {
  self.buffer
}

///|
pub fn Access::mode(self : Access) -> AccessMode {
  self.mode
}

///|
pub fn Collective::sequence(self : Collective) -> Int {
  self.sequence
}

///|
pub fn Collective::kind(self : Collective) -> CollectiveKind {
  self.kind
}

///|
pub fn Collective::element(self : Collective) -> ElementType {
  self.element
}

///|
pub fn Collective::send_bytes(self : Collective) -> Int64 {
  self.send_bytes
}

///|
pub fn Collective::receive_bytes(self : Collective) -> Int64 {
  self.receive_bytes
}

///|
pub fn Step::action(self : Step) -> Action {
  self.action
}

///|
pub fn Step::accesses(self : Step) -> ReadOnlyArray[Access] {
  self.accesses
}

///|
pub fn Step::dependencies(self : Step) -> ReadOnlyArray[Int] {
  self.dependencies
}

///|
pub fn Program::rank(self : Program) -> Int {
  self.rank
}

///|
pub fn Program::world_size(self : Program) -> Int {
  self.world_size
}

///|
pub fn Program::arenas(self : Program) -> ReadOnlyArray[Arena] {
  self.arenas
}

///|
pub fn Program::buffers(self : Program) -> ReadOnlyArray[Buffer] {
  self.buffers
}

///|
pub fn Program::steps(self : Program) -> ReadOnlyArray[Step] {
  self.steps
}

///|
pub fn Program::collective_steps(self : Program) -> ReadOnlyArray[Int] {
  self.collective_steps
}

///|
pub fn Program::lifetime(self : Program, buffer : Int) -> BufferLifetime? {
  self.lifetimes.get(buffer)
}

///|
pub fn BufferLifetime::users(self : BufferLifetime) -> ReadOnlyArray[Int] {
  self.users
}

///|
pub fn BufferLifetime::first_use(self : BufferLifetime) -> Int? {
  self.users.get(0)
}

///|
pub fn BufferLifetime::last_use(self : BufferLifetime) -> Int? {
  self.users.get(self.users.length() - 1)
}

///|
pub fn Schedule::program(self : Schedule) -> Program {
  self.program
}

///|
pub fn Schedule::policy(self : Schedule) -> Policy {
  self.policy
}

///|
pub fn Schedule::commands(
  self : Schedule,
  lane : Lane,
) -> ReadOnlyArray[Command] {
  match lane {
    ComputeLane => self.compute
    CommunicationLane => self.communication
  }
}

///|
pub fn Schedule::event_producers(self : Schedule) -> ReadOnlyArray[Int] {
  self.event_producers
}

///|
/// All listed queue tails must complete before invocation retirement.
pub fn Schedule::completion_steps(self : Schedule) -> ReadOnlyArray[Int] {
  self.tails
}

///|
pub fn Group::ranks(self : Group) -> ReadOnlyArray[Program] {
  self.ranks
}
