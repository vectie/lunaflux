///|
pub fn LocalTopologyRank::ordinal(self : LocalTopologyRank) -> Int32 {
  self.ordinal
}

///|
pub fn LocalTopologyRank::name(self : LocalTopologyRank) -> String {
  self.name
}

///|
pub fn LocalTopologyRank::total_memory_bytes(self : LocalTopologyRank) -> Int64 {
  self.total_memory_bytes
}

///|
pub fn LocalTopologyRank::target(
  self : LocalTopologyRank,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LocalTopologyDeclaration::rank_count(
  self : LocalTopologyDeclaration,
) -> Int32 {
  self.ranks.length()
}

///|
pub fn LocalTopologyDeclaration::rank(
  self : LocalTopologyDeclaration,
  rank : Int32,
) -> LocalTopologyRank? {
  self.ranks.get(rank)
}

///|
pub fn LocalTopologyDeclaration::link_count(
  self : LocalTopologyDeclaration,
) -> Int32 {
  self.links.length()
}

///|
pub fn LocalTopologyProbeProjection::device_count(
  self : LocalTopologyProbeProjection,
) -> Int32 {
  self.devices.length()
}

///|
pub fn LocalTopologyProbeProjection::peer_access_count(
  self : LocalTopologyProbeProjection,
) -> Int32 {
  self.peer_access.length()
}

///|
pub fn AdmittedLocalTopology::driver_version(
  self : AdmittedLocalTopology,
) -> Int32 {
  self.driver_version
}

///|
pub fn AdmittedLocalTopology::world_size(self : AdmittedLocalTopology) -> Int32 {
  self.ranks.length()
}

///|
pub fn AdmittedLocalTopology::rank(
  self : AdmittedLocalTopology,
  rank : Int32,
) -> LocalTopologyRank? {
  self.ranks.get(rank)
}

///|
pub fn AdmittedLocalTopology::target(
  self : AdmittedLocalTopology,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdmittedLocalTopology::link_count(self : AdmittedLocalTopology) -> Int32 {
  self.link_count
}
