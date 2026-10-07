///|
pub(all) enum OperandReadAction {
  ReadOperand(item~ : Int, slot~ : Int)
  ConsumeOperand(item~ : Int, slot~ : Int)
} derive(Eq, Debug)

///|
pub extend OperandReadAction with Eq::{equal, not_equal}

///|
pub extend OperandReadAction with @debug.Debug::{to_repr}

///|
/// Read-only operands in one published epoch. Read-ahead fills bounded slots;
/// a slot may be rebound only after its previous value is consumed. Consumers
/// remain ascending, so this never reassociates any accumulator's reduction.
pub struct OrderedOperandReads {
  priv items : Int
  priv slots : Int
} derive(Eq, Debug)

///|
pub extend OrderedOperandReads with Eq::{equal, not_equal}

///|
pub extend OrderedOperandReads with @debug.Debug::{to_repr}

///|
pub fn OrderedOperandReads::new(
  items~ : Int,
  slots~ : Int,
) -> OrderedOperandReads? {
  guard items > 0 && items <= 1024 && slots > 0 && slots <= items else {
    return None
  }
  Some({ items, slots, })
}

///|
pub fn OrderedOperandReads::geometry(self : OrderedOperandReads) -> (Int, Int) {
  (self.items, self.slots)
}

///|
pub fn OrderedOperandReads::actions(
  self : OrderedOperandReads,
) -> ReadOnlyArray[OperandReadAction] {
  let actions : Array[OperandReadAction] = []
  for item in 0..<self.slots {
    actions.push(ReadOperand(item~, slot=item))
  }
  for item in 0..<self.items {
    actions.push(ConsumeOperand(item~, slot=item % self.slots))
    if item + self.slots < self.items {
      actions.push(ReadOperand(item=item + self.slots, slot=item % self.slots))
    }
  }
  ReadOnlyArray::from_array(actions)
}

///|
pub fn OrderedOperandReads::canonical(self : OrderedOperandReads) -> String {
  "ordered-operand-reads-v1;items=\{self.items};slots=\{self.slots};epoch=published;consume=ascending"
}
