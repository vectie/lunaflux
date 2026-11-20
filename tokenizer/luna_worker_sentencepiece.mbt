///|
fn LunaTokenizerWorker::sentencepiece_invalid_utf8(
  self : LunaTokenizerWorker,
) -> Int32 raise TokenizerError {
  let byte_offset = self.input_offset
  self.clear_request()
  raise InvalidUtf8Input(byte_offset~)
}

///|
fn LunaTokenizerWorker::sentencepiece_scalar_width(
  self : LunaTokenizerWorker,
) -> Int32 raise TokenizerError {
  let available = self.input_length - self.input_offset
  if available <= 0 {
    ignore(self.sentencepiece_invalid_utf8())
  }
  let first = self.input[self.input_offset].to_int()
  if first <= 0x7f {
    return 1
  }
  let width = if first >= 0xc2 && first <= 0xdf {
    2
  } else if first >= 0xe0 && first <= 0xef {
    3
  } else if first >= 0xf0 && first <= 0xf4 {
    4
  } else {
    self.sentencepiece_invalid_utf8()
  }
  if width > available {
    ignore(self.sentencepiece_invalid_utf8())
  }
  for index in 1..<width {
    let value = self.input[self.input_offset + index].to_int()
    if value < 0x80 || value > 0xbf {
      ignore(self.sentencepiece_invalid_utf8())
    }
  }
  let second = self.input[self.input_offset + 1].to_int()
  if (first == 0xe0 && second < 0xa0) ||
    (first == 0xed && second > 0x9f) ||
    (first == 0xf0 && second < 0x90) ||
    (first == 0xf4 && second > 0x8f) {
    ignore(self.sentencepiece_invalid_utf8())
  }
  width
}

///|
fn sentencepiece_meta_byte(index : Int32) -> Byte {
  match index {
    0 => SENTENCEPIECE_META_SPACE_0
    1 => SENTENCEPIECE_META_SPACE_1
    2 => SENTENCEPIECE_META_SPACE_2
    _ => abort("SentencePiece meta-space byte invariant")
  }
}

///|
fn LunaTokenizerWorker::compare_sentencepiece_scalar(
  self : LunaTokenizerWorker,
  token_id : Int32,
  width : Int32,
) -> Int32 {
  let piece = self.spec.tokens_by_id[token_id].piece
  let common = Int32::min(width, piece.length())
  for index in 0..<common {
    let comparison = self.input[self.input_offset + index]
      .to_int()
      .compare(piece[index].to_int())
    if comparison != 0 {
      return comparison
    }
  }
  width.compare(piece.length())
}

///|
fn LunaTokenizerWorker::find_sentencepiece_scalar(
  self : LunaTokenizerWorker,
  width : Int32,
) -> Int32 {
  let candidates = self.spec.initial_scalar_ids
  for low = 0, high = candidates.length(); low < high; {
    let middle = low + (high - low) / 2
    let comparison = self.compare_sentencepiece_scalar(
      candidates[middle],
      width,
    )
    if comparison < 0 {
      continue low, middle
    } else if comparison > 0 {
      continue middle + 1, high
    } else {
      break candidates[middle]
    }
  } nobreak {
    -1
  }
}

///|
fn LunaTokenizerWorker::start_sentencepiece_scalar(
  self : LunaTokenizerWorker,
  meta : Bool,
  source_advance : Int32,
  byte_length : Int32,
) -> Unit {
  self.scalar_is_meta = meta
  self.scalar_direct_token_id = if meta {
    self.spec.sentencepiece_meta_token_id
  } else {
    self.find_sentencepiece_scalar(byte_length)
  }
  self.scalar_byte_offset = 0
  self.scalar_byte_length = byte_length
  self.scalar_source_advance = source_advance
  self.phase = LUNA_WORKER_SENTENCEPIECE_SCALAR
}

///|
fn LunaTokenizerWorker::finish_sentencepiece_scalar(
  self : LunaTokenizerWorker,
) -> Unit {
  self.segment_start = false
  if self.scalar_source_advance > 0 {
    self.input_offset += self.scalar_source_advance
    self.ordinary_ready = false
  }
  self.phase = LUNA_WORKER_INITIALIZE
}

///|
fn LunaTokenizerWorker::progress_sentencepiece_scalar(
  self : LunaTokenizerWorker,
) -> Unit {
  if self.scalar_direct_token_id >= 0 {
    self.append_symbol(self.scalar_direct_token_id)
    self.scalar_direct_token_id = -1
    self.finish_sentencepiece_scalar()
    return
  }
  if self.scalar_byte_offset >= self.scalar_byte_length {
    self.finish_sentencepiece_scalar()
    return
  }
  let byte = if self.scalar_is_meta {
    sentencepiece_meta_byte(self.scalar_byte_offset)
  } else {
    self.input[self.input_offset + self.scalar_byte_offset]
  }
  self.append_symbol(self.spec.byte_token_ids[byte.to_int()])
  self.scalar_byte_offset += 1
  if self.scalar_byte_offset >= self.scalar_byte_length {
    self.finish_sentencepiece_scalar()
  }
}
