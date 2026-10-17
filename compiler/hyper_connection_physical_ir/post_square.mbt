///|
/// Adjacent residual publication and successor square reduction. Publication
/// still rounds to BF16 before the square; fusion changes ownership, not math.
pub struct PostSquareFusion {
  priv streams : Int
  priv hidden : Int
  priv rows : Int
  priv workers : Int
}

///|
pub fn PostSquareFusion::new(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
) -> Self raise @precision.PrecisionPlanError {
  guard producer.geometry() == consumer.geometry() &&
    producer.mix_law() is RoundedBf16ControlsAndProducts &&
    consumer.control_square_law()
    is VectorizedTreeFloat32(partitions~, vector_width=1) &&
    consumer.control_law() is NormalizedBf16Projection else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  ignore(ParallelControlPlan::new(consumer))
  let (streams, hidden, rows) = consumer.geometry()
  { streams, hidden, rows, workers: partitions, }
}

///|
pub fn PostSquareFusion::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.streams, self.hidden, self.rows, self.workers)
}
