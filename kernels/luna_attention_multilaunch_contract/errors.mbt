///|
pub(all) enum AttentionMultiLaunchContractRule {
  Limits
  Operation
  Target
  Workspace
  Launch
  Module
  Symbol
  Operand
} derive(Debug, Eq)

///|
pub(all) suberror AttentionMultiLaunchContractError {
  InvalidAttentionMultiLaunchContract(AttentionMultiLaunchContractRule)
  MismatchedAttentionMultiLaunchContract(AttentionMultiLaunchContractRule)
} derive(Debug, Eq)
