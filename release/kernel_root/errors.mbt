///|
pub(all) enum LunaKernelRootRule {
  Limits
  Manifest
  Module
  ModuleOrdering
  Inventory
  Evidence
} derive(Debug, Eq)

///|
pub(all) suberror LunaKernelRootError {
  Invalid(LunaKernelRootRule)
  LimitExceeded(LunaKernelRootRule, maximum~ : Int64)
} derive(Debug, Eq)
