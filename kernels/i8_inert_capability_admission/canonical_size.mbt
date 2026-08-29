///|
const I8_ADMISSION_INTEGER_BYTES : Int = 8

///|
const I8_ADMISSION_SHA256_HEX_BYTES : Int = 64

///|
const I8_ADMISSION_SHA256_TEXT_BYTES : Int = 72

///|
const I8_ADMISSION_V1_DOMAIN_BYTES : Int = 49

///|
const I8_ADMISSION_V1_BASE_BYTES : Int = 393

///|
const I8_ADMISSION_V1_OPERATION_BYTES : Int = 216

///|
const I8_ADMISSION_V1_WEIGHT_BYTES : Int = 224

///|
const I8_ADMISSION_MAX_OPERATIONS : Int = 65536

///|
const I8_ADMISSION_MAX_WEIGHTS : Int = 299590

///|
const I8_ADMISSION_MAX_CANONICAL_BYTES : Int = 64 * 1024 * 1024

///|
fn i8_admission_v1_constants_are_consistent() -> Bool {
  let shape_rank_two = I8_ADMISSION_INTEGER_BYTES * 4
  let shape_rank_one = I8_ADMISSION_INTEGER_BYTES * 3
  let entry_point = I8_ADMISSION_SHA256_TEXT_BYTES +
    I8_ADMISSION_INTEGER_BYTES * 2
  let base = I8_ADMISSION_V1_DOMAIN_BYTES +
    I8_ADMISSION_INTEGER_BYTES +
    I8_ADMISSION_SHA256_TEXT_BYTES * 4 +
    I8_ADMISSION_INTEGER_BYTES * 4 +
    I8_ADMISSION_INTEGER_BYTES +
    I8_ADMISSION_INTEGER_BYTES
  let operation = I8_ADMISSION_INTEGER_BYTES * 3 +
    I8_ADMISSION_SHA256_TEXT_BYTES +
    I8_ADMISSION_INTEGER_BYTES +
    entry_point +
    I8_ADMISSION_INTEGER_BYTES * 2 +
    I8_ADMISSION_INTEGER_BYTES
  let weight = I8_ADMISSION_INTEGER_BYTES +
    I8_ADMISSION_SHA256_TEXT_BYTES +
    shape_rank_two +
    I8_ADMISSION_INTEGER_BYTES +
    I8_ADMISSION_SHA256_TEXT_BYTES +
    shape_rank_one +
    I8_ADMISSION_INTEGER_BYTES
  base == I8_ADMISSION_V1_BASE_BYTES &&
  operation == I8_ADMISSION_V1_OPERATION_BYTES &&
  weight == I8_ADMISSION_V1_WEIGHT_BYTES
}

///|
/// Returns the exact v1 size only when every intermediate fits the supplied
/// envelope. Division-before-multiplication keeps hostile counts from wrapping.
fn i8_admission_v1_size_within(
  operation_count : Int,
  weight_count : Int,
  maximum : Int,
) -> Int? {
  if !i8_admission_v1_constants_are_consistent() ||
    operation_count < 0 ||
    weight_count < 0 ||
    maximum < I8_ADMISSION_V1_BASE_BYTES {
    return None
  }
  let mut remaining = maximum - I8_ADMISSION_V1_BASE_BYTES
  if operation_count > remaining / I8_ADMISSION_V1_OPERATION_BYTES {
    return None
  }
  let operation_bytes = operation_count * I8_ADMISSION_V1_OPERATION_BYTES
  remaining -= operation_bytes
  if weight_count > remaining / I8_ADMISSION_V1_WEIGHT_BYTES {
    return None
  }
  Some(
    I8_ADMISSION_V1_BASE_BYTES +
    operation_bytes +
    weight_count * I8_ADMISSION_V1_WEIGHT_BYTES,
  )
}
