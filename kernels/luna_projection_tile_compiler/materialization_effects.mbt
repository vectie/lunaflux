///|
/// Visibility/reuse scope, not a device instruction or a fixed lane width.
pub enum ProjectionStorageFence {
  WorkgroupFence
  ConsumerGroupFence
} derive(Debug, Eq)

///|
/// Cooperation domain of a materialized value. A device backend chooses the
/// instruction; the compiler records which readers must observe publication.
pub(all) enum ProjectionMaterializationScope {
  ConsumerGroupStorage
  WorkgroupStorage
} derive(Debug, Eq)

///|
/// Publication is required even for a one-shot arena. Reader release is needed
/// only when a later iteration reuses that same storage; kernel exit ends a
/// one-shot lifetime without inserting a redundant device fence.
pub struct ProjectionMaterializationEffects {
  priv scope : ProjectionMaterializationScope
  priv reused : Bool
} derive(Debug, Eq)

///|
pub fn ProjectionMaterializationScope::materialization_effects(
  self : ProjectionMaterializationScope,
  reused~ : Bool,
) -> ProjectionMaterializationEffects {
  { scope: self, reused, }
}

///|
pub fn ProjectionMaterializationEffects::publication(
  self : ProjectionMaterializationEffects,
) -> ProjectionStorageFence {
  match self.scope {
    ConsumerGroupStorage => ConsumerGroupFence
    WorkgroupStorage => WorkgroupFence
  }
}

///|
pub fn ProjectionMaterializationEffects::release(
  self : ProjectionMaterializationEffects,
) -> ProjectionStorageFence? {
  if self.reused {
    Some(self.publication())
  } else {
    None
  }
}
