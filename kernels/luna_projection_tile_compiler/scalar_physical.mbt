///|
/// Physical product-of-folds without target instruction names. Every product
/// consumes operand 0 and its own right operand, and owns one accumulator.
/// Partitioning never changes the ordered multiply-then-add within a worker.
pub struct ProjectionScalarFoldPlan {
  priv extent : Int32
  priv products : Int32
  priv workers : Int32
} derive(Debug, Eq)

///|
pub extend ProjectionScalarFoldPlan with Eq::{equal, not_equal}

///|
pub extend ProjectionScalarFoldPlan with @debug.Debug::{to_repr}

///|
pub fn ProjectionScalarFoldPlan::new(
  extent~ : Int32,
  products~ : Int32,
  reduction : ProjectionSingleRowReduction,
) -> ProjectionScalarFoldPlan? {
  let workers = match reduction {
    OrderedDot => 1
    StridedPairwiseDot(lanes) => lanes
  }
  guard extent > 0 &&
    products > 0 &&
    products < 2147483647 &&
    workers > 0 &&
    (workers & (workers - 1)) == 0 else {
    return None
  }
  Some({ extent, products, workers, })
}

///|
pub fn ProjectionScalarFoldPlan::geometry(
  self : ProjectionScalarFoldPlan,
) -> (Int32, Int32, Int32) {
  (self.extent, self.products, self.workers)
}

///|
/// SSA operand IDs and destination accumulator for the selected product.
pub fn ProjectionScalarFoldPlan::product(
  self : ProjectionScalarFoldPlan,
  product : Int32,
) -> (Int32, Int32, Int32)? {
  if product < 0 || product >= self.products {
    None
  } else {
    Some((0, product + 1, product))
  }
}

///|
/// Every in-domain component belongs to exactly one worker/iteration pair.
pub fn ProjectionScalarFoldPlan::read_at(
  self : ProjectionScalarFoldPlan,
  worker~ : Int32,
  iteration~ : Int32,
) -> Int32? {
  guard worker >= 0 && worker < self.workers && iteration >= 0 else {
    return None
  }
  let offset = worker.to_int64() +
    iteration.to_int64() * self.workers.to_int64()
  if offset < self.extent.to_int64() {
    Some(offset.to_int())
  } else {
    None
  }
}

///|
/// Ordered descending tree edges; an empty list means no reassociation.
pub fn ProjectionScalarFoldPlan::combination_distances(
  self : ProjectionScalarFoldPlan,
) -> ReadOnlyArray[Int32] {
  let distances : Array[Int32] = []
  for distance = self.workers / 2; distance > 0; distance = distance / 2 {
    distances.push(distance)
  }
  ReadOnlyArray::from_array(distances)
}

///|
/// Source lane of one tree edge. Missing edges are outside the useful prefix;
/// only accumulator owner zero is eventually published.
pub fn ProjectionScalarFoldPlan::combine_from(
  self : ProjectionScalarFoldPlan,
  worker~ : Int32,
  distance~ : Int32,
) -> Int32? {
  guard worker >= 0 &&
    worker < self.workers &&
    distance > 0 &&
    distance < self.workers &&
    (distance & (distance - 1)) == 0 else {
    return None
  }
  if worker < self.workers - distance {
    Some(worker + distance)
  } else {
    None
  }
}
