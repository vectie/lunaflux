///|
#valtype
pub(all) enum Action {
  Reject
  BeginSpill
  BeginRestore
  PollTransfer
  CancelTransfer
  EvictHost
}

///|
/// Pure protocol state. The interpreter alone performs the selected effect.
#valtype
pub struct Control {
  worker : UInt64
  sequence : UInt64
  active : Bool
  original : Command
}

///|
#valtype
pub(all) struct Decision {
  next : Control
  action : Action
}

///|
pub fn Control::new(worker : UInt64) -> Control {
  {
    worker,
    sequence: 0UL,
    active: false,
    original: {
      worker,
      sequence: 0UL,
      opcode: Poll,
      page: 0,
      page_generation: 0,
      host: { slot: -1, generation: 0, },
    },
  }
}

///|
pub fn Control::decide(
  self : Control,
  command : Command,
  capability : Capability,
) -> Decision {
  let reject = { next: self, action: Reject, }
  if !command.valid(capability, self.worker) {
    return reject
  }
  match command.opcode {
    Poll | Cancel => {
      if !self.active || !command.same_transaction(self.original) {
        return reject
      }
      {
        next: self,
        action: if command.opcode is Poll {
          PollTransfer
        } else {
          CancelTransfer
        },
      }
    }
    _ => {
      if self.active ||
        self.sequence == 0xffffffffffffffffUL ||
        command.sequence != self.sequence + 1UL {
        return reject
      }
      let action = match command.opcode {
        Spill => BeginSpill
        Restore => BeginRestore
        Evict => EvictHost
        _ => Reject
      }
      {
        next: {
          ..self,
          sequence: command.sequence,
          active: !(command.opcode is Evict),
          original: command,
        },
        action,
      }
    }
  }
}

///|
pub fn Control::completed(self : Control) -> Control {
  { ..self, active: false, }
}
