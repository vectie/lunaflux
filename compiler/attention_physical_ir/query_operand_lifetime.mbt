///|
/// Joint lifetime refinement of the query/score/probability/value program.
/// Query operands are immutable across ascending KV epochs; score and PV
/// operands are epoch-local. Retention changes storage, not the fold law.
pub struct QueryOperandLifetime {
  priv program : OnlineFragmentProgram
  priv retained_dimensions : Int
  priv operand_words : Int
  priv accumulator_words : Int
  priv joint_right_consumers : Bool
} derive(Eq, Debug)

///|
pub extend QueryOperandLifetime with Eq::{equal, not_equal}

///|
pub extend QueryOperandLifetime with @debug.Debug::{to_repr}

///|
/// Instruction representation sizes are explicit refinement inputs. The
/// invariant prefix must contain complete reduction fragments; no masked or
/// independently normalized probability can be retained as a query operand.
pub fn QueryOperandLifetime::new(
  program : OnlineFragmentProgram,
  retained_dimensions~ : Int,
  operand_words~ : Int,
  accumulator_words~ : Int,
  joint_right_consumers? : Bool = false,
) -> QueryOperandLifetime? {
  let (_, _, dimensions) = program.fold().geometry()
  let (_, _, quantum) = program.geometry()
  guard retained_dimensions >= 0 &&
    retained_dimensions <= dimensions &&
    retained_dimensions % quantum == 0 &&
    operand_words > 0 &&
    operand_words <= 1024 &&
    accumulator_words > 0 &&
    accumulator_words <= 1024 &&
    (!joint_right_consumers || retained_dimensions > 0) else {
    return None
  }
  Some({
    program,
    retained_dimensions,
    operand_words,
    accumulator_words,
    joint_right_consumers,
  })
}

///|
/// All independent right-column products consume the same immutable query
/// representation in one instruction region. Their per-accumulator ordered
/// reduction remains unchanged; this refines consumption, not the fold law.
pub fn QueryOperandLifetime::joint_right_consumers(
  self : QueryOperandLifetime,
) -> Bool {
  self.joint_right_consumers
}

///|
pub fn QueryOperandLifetime::program(
  self : QueryOperandLifetime,
) -> OnlineFragmentProgram {
  self.program
}

///|
pub fn QueryOperandLifetime::retained_dimensions(
  self : QueryOperandLifetime,
) -> Int {
  self.retained_dimensions
}

///|
pub fn QueryOperandLifetime::retained_fragments(
  self : QueryOperandLifetime,
) -> Int {
  self.retained_dimensions / self.program.geometry().2
}

///|
/// Query-load count for an arbitrary number of KV epochs, including no-work
/// and ragged-tail folds. Query masks do not change the complete-fragment ABI.
pub fn QueryOperandLifetime::query_load_count(
  self : QueryOperandLifetime,
  epochs : Int,
) -> Int64? {
  guard epochs >= 0 else { return None }
  let (_, _, dimensions) = self.program.fold().geometry()
  let fragments = dimensions / self.program.geometry().2
  let retained = self.retained_fragments()
  Some(
    if epochs == 0 {
      0L
    } else {
      retained.to_int64() +
      (fragments - retained).to_int64() * epochs.to_int64()
    },
  )
}

///|
/// Peak scalar storage for fragment state, excluding scheduling/control
/// temporaries. This is not a claim about compiled hardware register count.
/// Output lives through both phases; score and probability have disjoint uses.
pub fn QueryOperandLifetime::peak_fragment_words(
  self : QueryOperandLifetime,
) -> Int {
  let (_, keys, dimensions) = self.program.fold().geometry()
  let (_, columns, _) = self.program.geometry()
  let retained = self.retained_fragments() * self.operand_words
  let output = dimensions / columns * self.accumulator_words
  let score = keys / columns * self.accumulator_words
  let transient_query = if self.retained_dimensions < dimensions {
    self.operand_words
  } else {
    0
  }
  // PV forwards one probability and one RHS fragment at a time, with no
  // simultaneous query reload. The score accumulator still owns probabilities.
  let pv = self.operand_words * 2 + self.accumulator_words
  let qk = transient_query + self.operand_words
  retained + output + score + Int::max(qk, pv)
}
