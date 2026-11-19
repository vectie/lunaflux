///|
pub(all) suberror ParallelError {
  Capacity
  Buffer(Int32)
  Step(Int32)
  Dependency(Int32)
  Uninitialized(step~ : Int32, buffer~ : Int32)
  CollectiveOrder(Int32)
  Group
}

///|
pub(all) enum AccessMode {
  Read
  Write
  ReadWrite
} derive(Eq)

///|
pub extend AccessMode with Eq::{equal, not_equal}

///|
pub(all) enum CollectiveKind {
  SumAllReduce
  LastAxisAllGather
} derive(Eq)

///|
pub extend CollectiveKind with Eq::{equal, not_equal}

///|
pub(all) enum ElementType {
  BFloat16
  Float16
  Float32
} derive(Eq)

///|
pub extend ElementType with Eq::{equal, not_equal}

///|
pub struct Arena {
  priv bytes : Int64
  priv initialized : Bool
}

///|
pub struct Buffer {
  priv arena : Int32
  priv offset : Int64
  priv bytes : Int64
}

///|
pub struct Access {
  priv buffer : Int32
  priv mode : AccessMode
}

///|
pub struct Collective {
  priv sequence : Int32
  priv kind : CollectiveKind
  priv element : ElementType
  priv send_bytes : Int64
  priv receive_bytes : Int64
}

///|
pub(all) enum Action {
  Compute(Int32)
  Communicate(Collective)
}

///|
pub struct Step {
  priv action : Action
  priv accesses : ReadOnlyArray[Access]
  priv dependencies : ReadOnlyArray[Int32]
}

///|
pub struct BufferLifetime {
  priv users : ReadOnlyArray[Int32]
}

///|
/// Inert, immutable physical effect graph. Construction grants no execution authority.
pub struct Program {
  priv rank : Int32
  priv world_size : Int32
  priv arenas : ReadOnlyArray[Arena]
  priv buffers : ReadOnlyArray[Buffer]
  priv steps : ReadOnlyArray[Step]
  priv lifetimes : ReadOnlyArray[BufferLifetime]
  priv collective_steps : ReadOnlyArray[Int32]
}

///|
pub(all) enum Policy {
  Ordered
  ComputeCommunicationOverlap
} derive(Eq)

///|
pub extend Policy with Eq::{equal, not_equal}

///|
pub(all) enum Lane {
  ComputeLane
  CommunicationLane
} derive(Eq)

///|
pub extend Lane with Eq::{equal, not_equal}

///|
pub(all) enum Command {
  Execute(step~ : Int32)
  Record(event~ : Int32)
  Wait(event~ : Int32)
} derive(Eq)

///|
pub extend Command with Eq::{equal, not_equal}

///|
pub struct Schedule {
  priv program : Program
  priv policy : Policy
  priv compute : ReadOnlyArray[Command]
  priv communication : ReadOnlyArray[Command]
  priv event_producers : ReadOnlyArray[Int32]
  priv tails : ReadOnlyArray[Int32]
}

///|
pub struct Group {
  priv ranks : ReadOnlyArray[Program]
}
