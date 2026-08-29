///|
fn read_numeric_exact(
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
  scratch : FixedArray[Byte],
  position : Int64,
  byte_count : Int,
) -> Unit raise NumericWeightFileError {
  if byte_count < 0 || byte_count > scratch.length() {
    raise FileTruncated
  }
  read(scratch, 0, position, byte_count)
}

///|
fn decode_f32_le(source : FixedArray[Byte], offset : Int) -> (UInt, Float) {
  let bits = source[offset].to_uint() |
    (source[offset + 1].to_uint() << 8) |
    (source[offset + 2].to_uint() << 16) |
    (source[offset + 3].to_uint() << 24)
  (bits, Float::reinterpret_from_uint(bits))
}

///|
fn validate_scale_region(
  region : NumericTransferRegion,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Unit raise NumericWeightFileError {
  if region.byte_length % 4L != 0L {
    raise InvalidScale(ordinal=region.ordinal, row=0)
  }
  let mut position = 0L
  let mut row = 0
  while position < region.byte_length {
    let remaining = region.byte_length - position
    let available = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    let count = available - available % 4
    if count <= 0 {
      raise InvalidScale(ordinal=region.ordinal, row~)
    }
    read_numeric_exact(
      read,
      scratch,
      region.source_offset_bytes + position,
      count,
    )
    for offset = 0; offset < count; offset = offset + 4 {
      let (_, value) = decode_f32_le(scratch, offset)
      if value.is_nan() || value.is_inf() || value <= (0.0 : Float) {
        raise InvalidScale(ordinal=region.ordinal, row~)
      }
      row += 1
    }
    position += count.to_int64()
  }
}

///|
fn row_contains_nonzero_i8(
  region : NumericTransferRegion,
  row : Int64,
  row_bytes : Int64,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Bool raise NumericWeightFileError {
  let mut offset = 0L
  let mut nonzero = false
  while offset < row_bytes {
    let remaining = row_bytes - offset
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    read_numeric_exact(
      read,
      scratch,
      region.source_offset_bytes + row * row_bytes + offset,
      count,
    )
    for index in 0..<count {
      if scratch[index] == b'\x80' {
        raise ReservedI8Code(ordinal=region.ordinal, row=row.to_int())
      }
      if scratch[index] != b'\x00' {
        nonzero = true
      }
    }
    offset += count.to_int64()
  }
  nonzero
}

///|
fn validate_i8_scale_rows(
  weight : NumericTransferRegion,
  scale : NumericTransferRegion,
  row_count : Int64,
  row_bytes : Int64,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Unit raise NumericWeightFileError {
  if scale.byte_length != row_count * 4L {
    raise InvalidScale(ordinal=scale.ordinal, row=0)
  }
  if row_bytes > chunk_bytes.to_int64() {
    for row in 0L..<row_count {
      let nonzero = row_contains_nonzero_i8(
        weight, row, row_bytes, scratch, chunk_bytes, read,
      )
      read_numeric_exact(read, scratch, scale.source_offset_bytes + row * 4L, 4)
      let (bits, value) = decode_f32_le(scratch, 0)
      if value.is_nan() || value.is_inf() || value <= (0.0 : Float) {
        raise InvalidScale(ordinal=scale.ordinal, row=row.to_int())
      }
      if !nonzero && bits != 0x3f800000U {
        raise ZeroRowScaleMismatch(ordinal=weight.ordinal, row=row.to_int())
      }
    }
    return
  }
  let rows_by_weight = chunk_bytes.to_int64() / row_bytes
  let rows_by_scale = chunk_bytes / 4
  let batch_capacity = rows_by_weight.min(rows_by_scale.to_int64()).to_int()
  let zero_rows = FixedArray::make(batch_capacity, true)
  let mut first_row = 0L
  while first_row < row_count {
    let remaining_rows = row_count - first_row
    let batch_rows = if remaining_rows < batch_capacity.to_int64() {
      remaining_rows.to_int()
    } else {
      batch_capacity
    }
    let weight_bytes = row_bytes * batch_rows.to_int64()
    read_numeric_exact(
      read,
      scratch,
      weight.source_offset_bytes + first_row * row_bytes,
      weight_bytes.to_int(),
    )
    for batch_row in 0..<batch_rows {
      zero_rows[batch_row] = true
      let start = batch_row.to_int64() * row_bytes
      for offset in 0L..<row_bytes {
        let byte = scratch[(start + offset).to_int()]
        if byte == b'\x80' {
          raise ReservedI8Code(
            ordinal=weight.ordinal,
            row=(first_row + batch_row.to_int64()).to_int(),
          )
        }
        if byte != b'\x00' {
          zero_rows[batch_row] = false
        }
      }
    }
    read_numeric_exact(
      read,
      scratch,
      scale.source_offset_bytes + first_row * 4L,
      batch_rows * 4,
    )
    for batch_row in 0..<batch_rows {
      let row = first_row + batch_row.to_int64()
      let (bits, value) = decode_f32_le(scratch, batch_row * 4)
      if value.is_nan() || value.is_inf() || value <= (0.0 : Float) {
        raise InvalidScale(ordinal=scale.ordinal, row=row.to_int())
      }
      if zero_rows[batch_row] && bits != 0x3f800000U {
        raise ZeroRowScaleMismatch(ordinal=weight.ordinal, row=row.to_int())
      }
    }
    first_row += batch_rows.to_int64()
  }
}

///|
fn validate_plain_i8_region(
  region : NumericTransferRegion,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Unit raise NumericWeightFileError {
  let mut position = 0L
  while position < region.byte_length {
    let remaining = region.byte_length - position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    read_numeric_exact(
      read,
      scratch,
      region.source_offset_bytes + position,
      count,
    )
    for index in 0..<count {
      if scratch[index] == b'\x80' {
        raise ReservedI8Code(ordinal=region.ordinal, row=0)
      }
    }
    position += count.to_int64()
  }
}

///|
fn validate_finite_fp8_region(
  region : NumericTransferRegion,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Unit raise NumericWeightFileError {
  let mut position = 0L
  while position < region.byte_length {
    let remaining = region.byte_length - position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    read_numeric_exact(
      read,
      scratch,
      region.source_offset_bytes + position,
      count,
    )
    for index in 0..<count {
      let code = scratch[index]
      // Finite E4M3FN uses the two sign variants of exponent=15,
      // mantissa=7 as NaN. Every other byte is a finite encoded value.
      if code == b'\x7f' || code == b'\xff' {
        raise NonFiniteFp8Code(
          ordinal=region.ordinal,
          element=position.to_int() + index,
        )
      }
    }
    position += count.to_int64()
  }
}

///|
fn validate_numeric_payload(
  model_plan : @plan.ModelPlan,
  regions : ReadOnlyArray[NumericTransferRegion],
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> Unit raise NumericWeightFileError {
  let tensors = model_plan.numeric_binding().schema().tensors()
  for ordinal, tensor in tensors {
    let region = regions[ordinal]
    if tensor.role() == @numeric_contract.NumericTensorRole::scale_metadata() {
      validate_scale_region(region, scratch, chunk_bytes, read)
    }
    let dtype = tensor.storage().dtype()
    if dtype == @numeric_contract.StorageDType::f8_e4m3_finite() {
      validate_finite_fp8_region(region, scratch, chunk_bytes, read)
    } else if dtype == @numeric_contract.StorageDType::i8() {
      match tensor.scale() {
        Some(scale_reference) => {
          guard tensor.shape().rank() == 2 &&
            tensor.shape().dimension(0) is Some(row_count) &&
            tensor.shape().dimension(1) is Some(row_bytes) else {
            raise InvalidTensorShape(ordinal~, axis=0)
          }
          let scale_ordinal = scale_reference.tensor().as_int()
          guard regions.get(scale_ordinal) is Some(scale_region) else {
            raise InvalidScale(ordinal=scale_ordinal, row=0)
          }
          validate_i8_scale_rows(
            region, scale_region, row_count, row_bytes, scratch, chunk_bytes, read,
          )
        }
        None => validate_plain_i8_region(region, scratch, chunk_bytes, read)
      }
    }
  }
}
