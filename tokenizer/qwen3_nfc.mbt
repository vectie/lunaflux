///|
const QWEN3_NFC_MAX_UTF8_EXPANSION : Int = 3

///|
fn qwen3_pack_scalar(scalar : Int) -> Int {
  (qwen3_canonical_combining_class(scalar) << 21) | scalar
}

///|
fn qwen3_packed_scalar(value : Int) -> Int {
  value & 0x1fffff
}

///|
fn qwen3_packed_combining_class(value : Int) -> Int {
  (value >> 21) & 0xff
}

///|
fn LunaTokenizerWorker::qwen3_append_decomposed_scalar(
  self : LunaTokenizerWorker,
  scalar : Int,
) -> Unit raise TokenizerError {
  let s_base = 0xac00
  let l_base = 0x1100
  let v_base = 0x1161
  let t_base = 0x11a7
  let l_count = 19
  let v_count = 21
  let t_count = 28
  let n_count = v_count * t_count
  let s_count = l_count * n_count
  if scalar >= s_base && scalar < s_base + s_count {
    let syllable = scalar - s_base
    self.qwen3_append_decomposed_scalar(l_base + syllable / n_count)
    self.qwen3_append_decomposed_scalar(v_base + syllable % n_count / t_count)
    let trailing = syllable % t_count
    if trailing != 0 {
      self.qwen3_append_decomposed_scalar(t_base + trailing)
    }
    return
  }
  match qwen3_canonical_decomposition(scalar) {
    Some((offset, length)) =>
      for index in 0..<length {
        self.qwen3_append_decomposed_scalar(
          qwen3_decomposition_scalar(offset + index),
        )
      }
    None => {
      if self.normalized_length >= self.tokens.length() {
        raise LunaInvalidWorkerCapacity
      }
      self.tokens[self.normalized_length] = qwen3_pack_scalar(scalar)
      self.normalized_length += 1
    }
  }
}

///|
fn LunaTokenizerWorker::qwen3_order_short_segment(
  self : LunaTokenizerWorker,
  start : Int,
  end : Int,
) -> Unit {
  let first_mark = start + 1
  for index in first_mark..<end {
    let value = self.tokens[index]
    let combining = qwen3_packed_combining_class(value)
    let mut destination = index
    while destination > start &&
          qwen3_packed_combining_class(self.tokens[destination - 1]) > combining {
      self.tokens[destination] = self.tokens[destination - 1]
      destination -= 1
    }
    self.tokens[destination] = value
  }
}

///|
fn LunaTokenizerWorker::qwen3_order_long_segment(
  self : LunaTokenizerWorker,
  start : Int,
  end : Int,
) -> Unit {
  let mut output = start
  let mut marks = start
  if qwen3_packed_combining_class(self.tokens[start]) == 0 {
    self.next[output] = self.tokens[start]
    output += 1
    marks += 1
  }
  for combining in qwen3_nonzero_combining_classes {
    for index in marks..<end {
      if qwen3_packed_combining_class(self.tokens[index]) == combining {
        self.next[output] = self.tokens[index]
        output += 1
      }
    }
  }
  if output != end {
    abort("Unicode combining-class table is incomplete")
  }
  for index in start..<end {
    self.tokens[index] = self.next[index]
  }
}

///|
fn LunaTokenizerWorker::qwen3_order_segment(
  self : LunaTokenizerWorker,
  start : Int,
  end : Int,
) -> Unit {
  if end - start <= 16 {
    self.qwen3_order_short_segment(start, end)
  } else {
    self.qwen3_order_long_segment(start, end)
  }
}

///|
fn LunaTokenizerWorker::qwen3_canonical_order(
  self : LunaTokenizerWorker,
) -> Unit {
  if self.normalized_length <= 1 {
    return
  }
  let mut segment_start = 0
  for index in 1..<self.normalized_length {
    if qwen3_packed_combining_class(self.tokens[index]) == 0 {
      self.qwen3_order_segment(segment_start, index)
      segment_start = index
    }
  }
  self.qwen3_order_segment(segment_start, self.normalized_length)
}

///|
fn LunaTokenizerWorker::qwen3_compose_ordered(
  self : LunaTokenizerWorker,
) -> Unit {
  let mut output = 0
  let mut starter_position = -1
  let mut starter = 0
  let mut last_combining = 0
  for index in 0..<self.normalized_length {
    let packed = self.tokens[index]
    let scalar = qwen3_packed_scalar(packed)
    let combining = qwen3_packed_combining_class(packed)
    let composite = if starter_position >= 0 &&
      (last_combining < combining || last_combining == 0) {
      qwen3_canonical_composition(starter, scalar)
    } else {
      None
    }
    match composite {
      Some(value) => {
        self.tokens[starter_position] = qwen3_pack_scalar(value)
        starter = value
      }
      None => {
        self.tokens[output] = packed
        if combining == 0 {
          starter_position = output
          starter = scalar
        }
        last_combining = combining
        output += 1
      }
    }
  }
  self.normalized_length = output
}

///|
fn LunaTokenizerWorker::qwen3_write_normalized_scalar(
  self : LunaTokenizerWorker,
  scalar : Int,
) -> Unit {
  let offset = self.input_length
  if scalar <= 0x7f {
    self.input[offset] = scalar.to_byte()
    self.input_length += 1
  } else if scalar <= 0x7ff {
    self.input[offset] = (0xc0 | (scalar >> 6)).to_byte()
    self.input[offset + 1] = (0x80 | (scalar & 0x3f)).to_byte()
    self.input_length += 2
  } else if scalar <= 0xffff {
    self.input[offset] = (0xe0 | (scalar >> 12)).to_byte()
    self.input[offset + 1] = (0x80 | ((scalar >> 6) & 0x3f)).to_byte()
    self.input[offset + 2] = (0x80 | (scalar & 0x3f)).to_byte()
    self.input_length += 3
  } else {
    self.input[offset] = (0xf0 | (scalar >> 18)).to_byte()
    self.input[offset + 1] = (0x80 | ((scalar >> 12) & 0x3f)).to_byte()
    self.input[offset + 2] = (0x80 | ((scalar >> 6) & 0x3f)).to_byte()
    self.input[offset + 3] = (0x80 | (scalar & 0x3f)).to_byte()
    self.input_length += 4
  }
}

///|
fn LunaTokenizerWorker::qwen3_finish_nfc(self : LunaTokenizerWorker) -> Unit {
  self.qwen3_canonical_order()
  self.qwen3_compose_ordered()
  let scalar_count = self.normalized_length
  self.input_length = 0
  for index in 0..<scalar_count {
    self.qwen3_write_normalized_scalar(qwen3_packed_scalar(self.tokens[index]))
  }
  self.normalized_length = 0
  self.input_offset = 0
  self.qwen_segment_end = -1
  self.phase = LUNA_WORKER_INITIALIZE
}

///|
/// Progresses one source scalar through Unicode 17.0 canonical decomposition.
/// Final ordering/composition uses only worker-owned fixed scratch arrays.
fn LunaTokenizerWorker::progress_qwen3_normalize(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.input_offset >= self.input_length {
    self.qwen3_finish_nfc()
    return
  }
  let (scalar, width) = qwen3_scalar_at(
    self.input,
    self.input_length,
    self.input_offset,
  )
  self.qwen3_append_decomposed_scalar(scalar)
  self.input_offset += width
}
