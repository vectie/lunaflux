///|
/// Conservative dead-suffix elimination for an ordered model graph. With no
/// externally demanded output, persistent-state writes remain observable roots.
/// Retaining their complete prefix also retains every activation dependency and
/// their original order; it does not assume associative floating-point folds.
/// The returned bound is a semantic operation count, not a backend launch count.
pub fn ModelPlan::observed_operation_prefix(
  self : ModelPlan,
  output_demand~ : Bool,
) -> Int {
  let operations = self.operations()
  if output_demand {
    return operations.length()
  }
  let mut count = 0
  for index, operation in operations {
    if operation.persistent_state() != NoPersistentState {
      count = index + 1
    }
  }
  count
}
