///|
/// Bounded topology validation area. No device name or vendor payload enters
/// an error.
pub(all) enum LocalTopologyArea {
  DeclarationRanks
  DeclarationLinks
  ProbeDevices
  ProbeLinks
  AdmissionRanks
  AdmissionLinks
} derive(Debug, Eq)

///|
/// Stable fail-closed topology rule.
pub(all) enum LocalTopologyRule {
  TooFew
  TooMany
  InvalidOrdinal
  InvalidName
  InvalidMemory
  RankOrder
  Duplicate
  Incomplete
  Unsupported
  HeterogeneousTarget
  ProbeUnavailable
  CountMismatch
  IdentityMismatch
  TargetMismatch
} derive(Debug, Eq)

///|
/// Payload-safe startup rejection with bounded rank/link indexes only.
pub(all) suberror LocalTopologyError {
  InvalidLocalTopology(
    area~ : LocalTopologyArea,
    rule~ : LocalTopologyRule,
    first~ : Int,
    second~ : Int
  )
} derive(Debug, Eq)

///|
fn invalid(
  area : LocalTopologyArea,
  rule : LocalTopologyRule,
  first : Int,
  second : Int,
) -> LocalTopologyError {
  InvalidLocalTopology(area~, rule~, first~, second~)
}
