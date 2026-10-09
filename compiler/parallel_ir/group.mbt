///|
/// Joins inert rank programs by collective sequence. Model/artifact identity
/// remains the responsibility of the physical-plan admission layer.
pub fn Group::new(ranks : ArrayView[Program]) -> Group raise ParallelError {
  if ranks.is_empty() {
    raise Group
  }
  let leader = ranks[0]
  for rank, program in ranks {
    if program.rank != rank ||
      program.world_size != ranks.length() ||
      program.collective_steps.length() != leader.collective_steps.length() {
      raise Group
    }
    for index, step in program.collective_steps {
      guard program.steps[step].action is Communicate(actual) else {
        raise Group
      }
      guard leader.steps[leader.collective_steps[index]].action
        is Communicate(expected) else {
        raise Group
      }
      if actual.sequence != expected.sequence ||
        actual.kind != expected.kind ||
        actual.element != expected.element ||
        actual.send_bytes != expected.send_bytes ||
        actual.receive_bytes != expected.receive_bytes {
        raise CollectiveOrder(index + 1)
      }
    }
  }
  { ranks: ReadOnlyArray::from_array(ranks), }
}
