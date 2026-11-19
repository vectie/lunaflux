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
  UnsupportedPlan(operation~ : Int32)
  UnsupportedBinding(reference~ : Int32)
  NonDivisible(LlamaTensorParallelDimension)
  InvalidAlignment
  ArithmeticOverflow(reference~ : Int32)
  ArenaTooLarge
  InvalidPhysicalPlan(@tensor_parallel_plan.TensorParallelPlanError)
} derive(Debug, Eq)
