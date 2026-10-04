///|
pub(all) enum SourcePlane {
  Payload
  Scales
} derive(Debug, Eq)

///|
pub extend SourcePlane with Eq::{equal, not_equal}

///|
pub extend SourcePlane with @debug.Debug::{to_repr}

///|
fn read_code(bytes : Bytes, offset : Int, width : Int) -> UInt {
  for i in 0..<width; code = 0U {
    continue code | (bytes[offset + i].to_uint() << (8 * i))
  } nobreak {
    code
  }
}

///|
/// Bounded startup-only streaming adapter. Readers refer to independently
/// authenticated immutable ranges; sinks own/upload the final BF16 arena.
/// No whole-tensor byte array, device readback, or token-step conversion.
fn stream_parameters(
  plan : @precision.OperandPlan,
  global_scale : Float,
  chunk_elements : Int,
  host_scratch_ceiling : Int64,
) -> (@format.Representation, Int) raise MaterializeError {
  if plan.role() != Weight ||
    plan.placement() != StartupExpandBf16 ||
    plan.effect() != StartupWrite ||
    chunk_elements <= 0 {
    raise WrongPlan
  }
  let rep = plan.representation()
  let (_, columns) = rep.shape()
  if global_scale.is_nan() ||
    global_scale.is_inf() ||
    global_scale <= 0.0F ||
    (!rep.has_global_scale() && global_scale != 1.0F) {
    raise Conversion(@format.PrecisionError::InvalidScale)
  }
  let chunk = chunk_elements.min(columns)
  let bits = rep.element().bits()
  let scale_width = rep.scale_element().bytes()
  // Row-local windows need at most one extra payload byte and one extra
  // scale cell. Two output buffers account for mutable scratch + upload Bytes.
  let scale_cells = match rep.grouping() {
    Absent => 0L
    Tensor | Row => 1L
    Block(columns=bc, ..) =>
      (chunk.to_int64() + bc.to_int64() - 1L) / bc.to_int64() + 1L
  }
  let ceiling = (chunk.to_int64() * bits.to_int64() + 7L) / 8L +
    1L +
    scale_cells * scale_width.to_int64() +
    chunk.to_int64() * 4L
  if ceiling > host_scratch_ceiling || chunk > 1073741823 {
    raise WrongPlan
  }
  (rep, chunk)
}

///|
pub fn stream_bf16_weight(
  plan : @precision.OperandPlan,
  global_scale? : Float = 1.0F,
  chunk_elements~ : Int,
  host_scratch_ceiling~ : Int64,
  read : (SourcePlane, Int64, Int) -> Bytes raise MaterializeError,
  consume : (Int64, Bytes) -> Unit raise MaterializeError,
) -> Unit raise MaterializeError {
  let (rep, chunk) = stream_parameters(
    plan, global_scale, chunk_elements, host_scratch_ceiling,
  )
  let (rows, columns) = rep.shape()
  let bits = rep.element().bits()
  let scale_width = rep.scale_element().bytes()
  let scratch = FixedArray::make(chunk * 2, b'\x00')
  for row in 0..<rows {
    for first = 0; first < columns; first = first + chunk.min(columns - first) {
      let count = chunk.min(columns - first)
      let begin = first.to_int64() * bits.to_int64() / 8L
      let end = ((first.to_int64() + count.to_int64()) * bits.to_int64() + 7L) /
        8L
      let payload = read(
        Payload,
        row.to_int64() * rep.row_bytes() + begin,
        (end - begin).to_int(),
      )
      if payload.length().to_int64() != end - begin {
        raise Conversion(@format.PrecisionError::PayloadMismatch)
      }
      if bits == 4 &&
        first + count == columns &&
        columns % 2 != 0 &&
        (payload[payload.length() - 1].to_uint() & 0xf0U) != 0U {
        raise Conversion(@format.PrecisionError::InvalidPadding)
      }
      let (base_scale, scales) = if scale_width == 0 {
        (0L, b"")
      } else {
        let a = rep.scale_index(row, first) catch { e => raise Conversion(e) }
        let b = rep.scale_index(row, first + count - 1) catch {
          e => raise Conversion(e)
        }
        let byte_count = (b - a + 1L) * scale_width.to_int64()
        if byte_count > 2147483647L {
          raise WrongPlan
        }
        let scales = read(
          Scales,
          a * scale_width.to_int64(),
          byte_count.to_int(),
        )
        if scales.length().to_int64() != byte_count {
          raise Conversion(@format.PrecisionError::PayloadMismatch)
        }
        (a, scales)
      }
      for i in 0..<count {
        let column = first + i
        let offset = (column.to_int64() * bits.to_int64() / 8L - begin).to_int()
        let code = if bits == 4 {
          (payload[offset].to_uint() >> (column % 2 * 4)) & 15U
        } else {
          read_code(payload, offset, bits / 8)
        }
        let value = @format.decode_element(rep.element(), code) catch {
          e => raise Conversion(e)
        }
        let scaled = if scale_width == 0 {
          value
        } else {
          let index = rep.scale_index(row, column) catch {
            e => raise Conversion(e)
          }
          let code = read_code(
            scales,
            ((index - base_scale) * scale_width.to_int64()).to_int(),
            scale_width,
          )
          let scale = @format.decode_scale(rep.scale_element(), code) catch {
            e => raise Conversion(e)
          }
          value * scale
        }
        let reconstructed = if rep.has_global_scale() {
          scaled * global_scale
        } else {
          scaled
        }
        let rounded = @format.encode_element(Bf16, reconstructed) catch {
          e => raise Conversion(e)
        }
        scratch[2 * i] = rounded.to_byte()
        scratch[2 * i + 1] = (rounded >> 8).to_byte()
      }
      let offset = (row.to_int64() * columns.to_int64() + first.to_int64()) * 2L
      consume(offset, Bytes::from_array(scratch[:count * 2]))
    }
  }
}

///|
/// Startup effect interpreter for the entire graph's weight schema. All
/// placement/global/budget checks precede the first source read or upload.
/// Source validation can still fail while streaming: the caller must keep the
/// unpublished arena private and release it on failure, then publish once.
pub fn stream_bf16_weights(
  schema : @precision.WeightSchema,
  global_scales : ArrayView[Float],
  chunk_elements~ : Int,
  host_scratch_ceiling~ : Int64,
  read : (Int, SourcePlane, Int64, Int) -> Bytes raise MaterializeError,
  consume : (Int64, Bytes) -> Unit raise MaterializeError,
) -> Unit raise MaterializeError {
  let bindings = schema.bindings()
  if bindings.length() != global_scales.length() {
    raise WrongPlan
  }
  for binding in bindings {
    ignore(
      stream_parameters(
        binding.operand(),
        global_scales[binding.ordinal()],
        chunk_elements,
        host_scratch_ceiling,
      ),
    )
  }
  for binding in bindings {
    stream_bf16_weight(
      binding.operand(),
      global_scale=global_scales[binding.ordinal()],
      chunk_elements~,
      host_scratch_ceiling~,
      (plane, offset, count) => read(binding.ordinal(), plane, offset, count),
      (offset, bytes) => consume(binding.offset() + offset, bytes),
    )
  }
}
