///|
pub(all) enum LlamaTensorParallelDimension {
  Vocabulary
  Hidden
  Intermediate
  AttentionHeads
  KeyValueHeads
} derive(Debug, Eq)

///|
pub(all) suberror LlamaTensorParallelError {
  InvalidTopology
  IdentityMismatch
  UnsupportedPlan(operation~ : Int)
  UnsupportedBinding(reference~ : Int)
  NonDivisible(LlamaTensorParallelDimension)
  InvalidAlignment
  ArithmeticOverflow(reference~ : Int)
  ArenaTooLarge
  InvalidPhysicalPlan(@tensor_parallel_plan.TensorParallelPlanError)
} derive(Debug, Eq)
