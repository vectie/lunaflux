///|
const NUMERIC_MAX_INT64 : Int64 = 9223372036854775807L

///|
/// Immutable checked dense shape and its exact element count.
///
/// Rank zero is the scalar shape and has one element. Every explicit dimension
/// must be positive. Construction owns the only shape-product arithmetic in
/// this package and rejects signed-64-bit overflow before multiplication.
pub struct NumericShape {
  priv dimensions : ReadOnlyArray[Int64]
  priv element_count : Int64
} derive(Debug, Eq)

///|
pub fn NumericShape::new(
  dimensions : ArrayView[Int64],
) -> NumericShape raise NumericContractError {
  if dimensions.length() > MODEL_NUMERIC_SCHEMA_HARD_MAX_RANK {
    raise NumericSchemaLimitExceeded(SchemaRank)
  }
  let copied : Array[Int64] = []
  let mut element_count = 1L
  for dimension in dimensions {
    if dimension <= 0L {
      raise InvalidNumericContract(ShapeDimension)
    }
    if element_count > NUMERIC_MAX_INT64 / dimension {
      raise InvalidNumericContract(ShapeArithmetic)
    }
    element_count = element_count * dimension
    copied.push(dimension)
  }
  { dimensions: ReadOnlyArray::from_array(copied), element_count }
}

///|
pub fn NumericShape::rank(self : NumericShape) -> Int {
  self.dimensions.length()
}

///|
pub fn NumericShape::dimension(self : NumericShape, index : Int) -> Int64? {
  self.dimensions.get(index)
}

///|
pub fn NumericShape::dimensions(self : NumericShape) -> ReadOnlyArray[Int64] {
  self.dimensions
}

///|
pub fn NumericShape::element_count(self : NumericShape) -> Int64 {
  self.element_count
}

///|
/// Returns the exact dense payload size for a checked shape and storage dtype.
///
/// This schema calculation neither performs conversion nor implies that a
/// device, kernel, or model family supports the dtype.
pub fn storage_byte_length(
  storage_dtype : StorageDType,
  shape : NumericShape,
) -> Int64 raise NumericContractError {
  let width = storage_dtype.byte_width()
  if width <= 0L || shape.element_count > NUMERIC_MAX_INT64 / width {
    raise InvalidNumericContract(StorageArithmetic)
  }
  shape.element_count * width
}
