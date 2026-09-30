///|
/// Producer coordinates factored into a workgroup-invariant row/vector map
/// and a varying ordered reduction offset. Only addresses are factored;
/// no operand read is moved across a publication or materialization effect.
pub struct ProjectionProducerAddressPlan {
  priv region : @physical_tile_ir.OperandRegion
  priv hoisted : Bool
} derive(Debug, Eq)

///|
pub extend ProjectionProducerAddressPlan with @debug.Debug::{to_repr}

///|
pub extend ProjectionProducerAddressPlan with Eq::{equal, not_equal}

///|
pub fn ProjectionPhysicalPipeline::producer_addresses(
  self : ProjectionPhysicalPipeline,
  operand : Int,
) -> ProjectionProducerAddressPlan? {
  self.operands
  .region(operand)
  .map(region => { region, hoisted: self.schedule.hoist_addresses, })
}

///|
pub fn ProjectionProducerAddressPlan::hoisted(
  self : ProjectionProducerAddressPlan,
) -> Bool {
  self.hoisted
}

///|
/// (vectors, workers, invariant address slots per worker).
pub fn ProjectionProducerAddressPlan::geometry(
  self : ProjectionProducerAddressPlan,
) -> (Int, Int, Int) {
  self.region.ownership().geometry()
}

///|
/// The canonical producer map is shared by every stage in the operand ring.
/// Returns row, reduction-vector offset and slot-relative shared offset.
pub fn ProjectionProducerAddressPlan::at(
  self : ProjectionProducerAddressPlan,
  worker : Int,
  address_slot : Int,
) -> (Int, Int, Int)? {
  self.region.copy_at(worker, address_slot)
}
