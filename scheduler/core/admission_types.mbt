///|
/// Opaque admission publication shell. The request and handle storage are
/// allocated before scheduler mutation; commit fills the exact handle using
/// scalar assignments only.
pub struct PreparedAdmission {
  priv owner : Scheduler
  priv request : TokenizedRequest
  priv handle : RequestHandle
  priv mut committed : Bool
  priv mut retention_issued : Bool
  priv mut publication_route : SchedulerPublicationRoute
}

///|
/// Boundary-restricted exclusive admission shell. All timeless validation and
/// owner allocation complete before a rooted worker is activated.
pub struct PreparedExclusiveAdmission {
  priv owner : Scheduler
  priv request : TokenizedRequest
  priv handle : RequestHandle
  priv mut committed : Bool
  priv mut consumed : Bool
  priv mut retention_issued : Bool
  priv mut publication_route : SchedulerPublicationRoute
}

///|
#valtype
pub(all) enum ExclusiveAdmissionCommit {
  ExclusiveAdmissionCommitted
  ExclusiveAdmissionExpired
  ExclusiveAdmissionInvalidated
} derive(Debug, Eq)
