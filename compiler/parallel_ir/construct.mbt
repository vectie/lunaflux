///|
pub fn Arena::new(
  bytes~ : Int64,
  initialized~ : Bool,
) -> Arena raise ParallelError {
  if bytes < 0L {
    raise Capacity
  }
  { bytes, initialized, }
}

///|
pub fn Buffer::new(
  arena~ : Int,
  offset_bytes~ : Int64,
  byte_count~ : Int64,
) -> Buffer raise ParallelError {
  if arena < 0 ||
    offset_bytes < 0L ||
    byte_count <= 0L ||
    offset_bytes > 9223372036854775807L - byte_count {
    raise Buffer(arena)
  }
  { arena, offset: offset_bytes, bytes: byte_count, }
}

///|
pub fn Access::new(
  buffer~ : Int,
  mode~ : AccessMode,
) -> Access raise ParallelError {
  if buffer < 0 {
    raise Buffer(buffer)
  }
  { buffer, mode, }
}

///|
pub fn Collective::new(
  sequence~ : Int,
  kind~ : CollectiveKind,
  element~ : ElementType,
  send_bytes~ : Int64,
  receive_bytes~ : Int64,
) -> Collective raise ParallelError {
  let width = match element {
    BFloat16 | Float16 => 2L
    Float32 => 4L
  }
  if sequence <= 0 ||
    send_bytes <= 0L ||
    receive_bytes <= 0L ||
    send_bytes % width != 0L ||
    receive_bytes % width != 0L {
    raise CollectiveOrder(sequence)
  }
  { sequence, kind, element, send_bytes, receive_bytes, }
}

///|
pub fn Step::new(
  action : Action,
  accesses : ArrayView[Access],
  dependencies? : ArrayView[Int] = [],
) -> Step {
  {
    action,
    accesses: ReadOnlyArray::from_array(accesses),
    dependencies: ReadOnlyArray::from_array(dependencies),
  }
}

///|
fn AccessMode::reads(self : AccessMode) -> Bool {
  self != Write
}

///|
fn AccessMode::writes(self : AccessMode) -> Bool {
  self != Read
}

///|
fn Collective::validate(
  self : Collective,
  expected : Int,
  world : Int,
) -> Unit raise ParallelError {
  if self.sequence != expected {
    raise CollectiveOrder(expected)
  }
  match self.kind {
    SumAllReduce =>
      if self.send_bytes != self.receive_bytes {
        raise CollectiveOrder(expected)
      }
    LastAxisAllGather =>
      if self.send_bytes > 9223372036854775807L / world.to_int64() ||
        self.receive_bytes != self.send_bytes * world.to_int64() {
        raise CollectiveOrder(expected)
      }
  }
}
