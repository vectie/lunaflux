///|
/// Offline ownership alternatives for a pointwise map of two ordered folds.
/// These choices change transport, not either reduction or the numeric map.
pub(all) enum ProjectionSiblingOwnership {
  SplitSiblingPlanes
  RetainSiblingProducer
  CoownedSiblingValues
} derive(Debug, Eq)

///|
pub extend ProjectionSiblingOwnership with @debug.Debug::{to_repr}

///|
pub extend ProjectionSiblingOwnership with Eq::{equal, not_equal}

///|
/// Immutable materialization contract. A retained producer keeps its values
/// local and imports only its peer; coowned values never enter shared storage.
pub struct ProjectionSiblingEpiloguePlan {
  priv ownership : ProjectionSiblingOwnership
  priv distribution : ProjectionProductFoldDistribution
  priv planes : Int
  priv elements : Int
  priv effects : ProjectionMaterializationEffects
} derive(Debug, Eq)

///|
pub extend ProjectionSiblingEpiloguePlan with @debug.Debug::{to_repr}

///|
pub extend ProjectionSiblingEpiloguePlan with Eq::{equal, not_equal}

///|
pub fn ProjectionPhysicalPipeline::sibling_epilogue(
  self : ProjectionPhysicalPipeline,
  ownership : ProjectionSiblingOwnership,
) -> ProjectionSiblingEpiloguePlan? {
  guard self.operands.region(2) is Some(_) else { return None }
  let ownership = if self.sibling_reuse {
    guard ownership != CoownedSiblingValues else { return None }
    ownership
  } else {
    CoownedSiblingValues
  }
  let rows = self.schedule.row_width()
  let columns = self.schedule.output_tiles() * self.schedule.tile_width()
  guard rows > 0 && columns > 0 && rows <= 2147483647 / columns else {
    return None
  }
  let planes = match ownership {
    SplitSiblingPlanes => 2
    RetainSiblingProducer => 1
    CoownedSiblingValues => 0
  }
  Some({
    ownership,
    distribution: self.distribution,
    planes,
    elements: rows * columns,
    effects: (if planes == 0 { ConsumerGroupStorage } else { WorkgroupStorage }).materialization_effects(
      reused=planes > 0,
    ),
  })
}

///|
pub fn ProjectionSiblingEpiloguePlan::ownership(
  self : ProjectionSiblingEpiloguePlan,
) -> ProjectionSiblingOwnership {
  self.ownership
}

///|
pub fn ProjectionSiblingEpiloguePlan::result_planes(
  self : ProjectionSiblingEpiloguePlan,
) -> Int {
  self.planes
}

///|
pub fn ProjectionSiblingEpiloguePlan::plane_elements(
  self : ProjectionSiblingEpiloguePlan,
) -> Int {
  self.elements
}

///|
pub fn ProjectionSiblingEpiloguePlan::distribution(
  self : ProjectionSiblingEpiloguePlan,
) -> ProjectionProductFoldDistribution {
  self.distribution
}

///|
pub fn ProjectionSiblingEpiloguePlan::effects(
  self : ProjectionSiblingEpiloguePlan,
) -> ProjectionMaterializationEffects? {
  if self.planes == 0 {
    None
  } else {
    Some(self.effects)
  }
}
