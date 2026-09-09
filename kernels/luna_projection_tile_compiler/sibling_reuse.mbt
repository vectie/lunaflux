///|
/// Distribute two independent sibling folds before distributing their rows.
/// The per-consumer accumulator product stays bounded, but each immutable
/// weight value can be reused across multiple rows. The reduction axis never
/// participates in this partition and therefore retains its original order.
pub fn partition_projection_sibling_reuse(
  row_tiles~ : Int,
  output_tiles~ : Int,
  consumer_groups~ : Int,
) -> ProjectionProductFoldDistribution? {
  guard output_tiles > 0 && consumer_groups > 0 && consumer_groups % 2 == 0 else {
    return None
  }
  // Bound the materialized sibling result to one tile per available consumer.
  guard row_tiles <= consumer_groups / output_tiles else { return None }
  guard partition_projection_product_fold(
      row_tiles~,
      output_tiles~,
      consumer_groups=consumer_groups / 2,
    )
    is Some(distribution) else {
    return None
  }
  // Splitting siblings is useful only when it creates actual weight reuse.
  guard distribution.rows_per_group() >= 2 else { return None }
  Some(distribution)
}
