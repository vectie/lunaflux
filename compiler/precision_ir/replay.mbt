///|
pub(all) enum ReplayScalar {
  ReplayF32
  ReplayBf16
  ReplayI32
}

///|
/// An already materialized semantic boundary, not a quality metric. None means
/// a caller-supplied input, Some(stage) a completed producer. Buffer and stage
/// indices are local to the owning effect program. Copies remain external.
pub struct NumericBoundary {
  label : String
  buffer : Int
  after_stage : Int?
  scalar : ReplayScalar
  capacity_bytes : Int64
  extent : NumericExtent
}

///|
/// A counted live prefix refers to an I32 field in another bound port, not to
/// capacity or an inferred batch. Read it only after the appropriate producer.
pub(all) enum NumericExtent {
  Capacity
  CountedRows(buffer~ : Int, index~ : Int, row_bytes~ : Int64)
}
