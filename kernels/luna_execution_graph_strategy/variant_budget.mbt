///|
/// Structural budget for optional graph variants. Counts executable graphs and
/// kernel nodes, not backend-private bytes. It does not replace a deployment's
/// device-memory ceiling. Reservation is a pure startup transition.
pub struct ExecutionGraphVariantBudget {
  priv remaining_graphs : Int
  priv remaining_nodes : Int
} derive(Debug, Eq)

///|
pub fn ExecutionGraphVariantBudget::new(
  maximum_graphs : Int,
  maximum_nodes : Int,
) -> ExecutionGraphVariantBudget raise ExecutionGraphBucketError {
  if maximum_graphs < 0 || maximum_nodes < 0 {
    raise InvalidExecutionGraphBucket(Capacity)
  }
  { remaining_graphs: maximum_graphs, remaining_nodes: maximum_nodes }
}

///|
pub fn ExecutionGraphVariantBudget::reserve(
  self : ExecutionGraphVariantBudget,
  nodes : Int,
) -> ExecutionGraphVariantBudget? {
  if nodes <= 0 || self.remaining_graphs == 0 || nodes > self.remaining_nodes {
    None
  } else {
    Some({
      remaining_graphs: self.remaining_graphs - 1,
      remaining_nodes: self.remaining_nodes - nodes,
    })
  }
}
