///|
/// Eliminates a single-use right-operand materialization inside one ordered
/// product. The producer's packed words feed the instruction operands by name;
/// the permutation is a renaming, never a copy or a numeric conversion.
/// Multiple left rows or lookahead slots require a different lifetime proof.
pub struct FragmentForwarding {
  priv program : FragmentProgram
  priv words : Int
  priv instruction_words : Int
  priv order : ReadOnlyArray[Int]
} derive(Eq, Debug)

///|
pub extend FragmentForwarding with Eq::{equal, not_equal}

///|
pub extend FragmentForwarding with @debug.Debug::{to_repr}

///|
pub fn FragmentProgram::forward_right(
  self : FragmentProgram,
  words~ : Int,
  instruction_words~ : Int,
  consumer_order : ArrayView[Int],
) -> FragmentForwarding? {
  let (rows, _, _, _, slots) = self.geometry()
  guard rows == 1 &&
    slots == 1 &&
    words > 0 &&
    words <= 64 &&
    instruction_words > 0 &&
    words % instruction_words == 0 &&
    consumer_order.length() == words else {
    return None
  }
  // A bijection preserves every packed lane, rejects aliasing and omissions,
  // and does not depend on the numerical payload of the fragment.
  for i, word in consumer_order {
    guard word >= 0 && word < words else { return None }
    for previous in 0..<i {
      guard consumer_order[previous] != word else { return None }
    }
  }
  Some({
    program: self,
    words,
    instruction_words,
    order: ReadOnlyArray::from_array(consumer_order),
  })
}

///|
pub fn FragmentForwarding::program(
  self : FragmentForwarding,
) -> FragmentProgram {
  self.program
}

///|
/// (packed producer words, words per consumer, ordered consumer count).
pub fn FragmentForwarding::geometry(
  self : FragmentForwarding,
) -> (Int, Int, Int) {
  (self.words, self.instruction_words, self.words / self.instruction_words)
}

///|
pub fn FragmentForwarding::word(
  self : FragmentForwarding,
  instruction : Int,
  operand : Int,
) -> Int? {
  guard instruction >= 0 &&
    instruction < self.words / self.instruction_words &&
    operand >= 0 &&
    operand < self.instruction_words else {
    return None
  }
  Some(self.order[instruction * self.instruction_words + operand])
}

///|
pub fn FragmentForwarding::canonical(self : FragmentForwarding) -> String {
  "fragment-forward-right-v1;program=\{self.program.canonical()};words=\{self.words};instruction-words=\{self.instruction_words};order=" +
  self.order.map(v => v.to_string()).join(",")
}
