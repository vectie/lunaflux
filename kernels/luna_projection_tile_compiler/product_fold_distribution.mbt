///|
/// Partition a product of independent ordered folds. Only the independent row
/// axis is distributed: no reduction term moves across another term, and no
/// accumulator is shared between consumers.
pub struct ProjectionProductFoldDistribution {
  priv row_groups : Int
  priv output_groups : Int
  priv rows_per_group : Int
} derive(Debug, Eq)

///|
pub fn partition_projection_product_fold(
  row_tiles~ : Int,
  output_tiles~ : Int,
  consumer_groups~ : Int,
) -> ProjectionProductFoldDistribution? {
  guard row_tiles > 0 && output_tiles > 0 && consumer_groups >= output_tiles else {
    return None
  }
  let capacity = Int::min(row_tiles, consumer_groups / output_tiles)
  // A divisor keeps every consumer's accumulator tuple statically sized.
  // Decreases: groups. Invariant: 1 <= groups <= capacity <= row_tiles.
  let row_groups = for groups = capacity; groups > 1; {
    if row_tiles % groups == 0 {
      break groups
    }
    continue groups - 1
  } nobreak {
    1
  }
  Some({
    row_groups,
    output_groups: output_tiles,
    rows_per_group: row_tiles / row_groups,
  })
}

///|
pub fn ProjectionProductFoldDistribution::row_groups(
  self : ProjectionProductFoldDistribution,
) -> Int {
  self.row_groups
}

///|
pub fn ProjectionProductFoldDistribution::output_groups(
  self : ProjectionProductFoldDistribution,
) -> Int {
  self.output_groups
}

///|
pub fn ProjectionProductFoldDistribution::rows_per_group(
  self : ProjectionProductFoldDistribution,
) -> Int {
  self.rows_per_group
}

///|
pub fn ProjectionProductFoldDistribution::consumer_groups(
  self : ProjectionProductFoldDistribution,
) -> Int {
  self.row_groups * self.output_groups
}
