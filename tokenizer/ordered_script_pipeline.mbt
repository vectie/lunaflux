///|
fn ordered_script_range(scalar : Int) -> Bool {
  scalar is (0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff)
}

///|
fn unicode_is_symbol(scalar : Int) -> Bool {
  // Inclusive pairs of 24-bit scalars from Unicode's P*/S* categories.
  for low = 0, high = unicode_symbol_ranges.length() / 6; low < high; {
    let middle = low + (high - low) / 2
    let first = qwen3_unicode_u24(unicode_symbol_ranges, middle * 6)
    let last = qwen3_unicode_u24(unicode_symbol_ranges, middle * 6 + 3)
    if scalar < first {
      continue low, middle
    }
    if scalar > last {
      continue middle + 1, high
    }
    break true
  } nobreak {
    false
  }
}

///|
fn ordered_word_scalar(scalar : Int, script_fragment : Bool) -> Bool {
  ordered_script_range(scalar) == script_fragment &&
  !qwen3_is_number(scalar) &&
  (qwen3_is_letter(scalar) || qwen35_is_mark(scalar))
}

///|
fn ordered_word_end(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
  script_fragment : Bool,
) -> Int raise TokenizerError {
  for cursor = offset; cursor < length; {
    let (scalar, width) = qwen3_scalar_at(input, length, cursor)
    if !ordered_word_scalar(scalar, script_fragment) {
      break cursor
    }
    continue cursor + width
  } nobreak {
    cursor
  }
}

///|
/// Match the third split at this scalar, without crossing either earlier split.
/// -1 means no match, so the caller preserves an unmatched run as one pretoken.
fn ordered_word_match_end(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
  script_fragment : Bool,
) -> Int raise TokenizerError {
  let (first, width) = qwen3_scalar_at(input, length, offset)
  if ordered_script_range(first) != script_fragment || qwen3_is_number(first) {
    return -1
  }
  if first is (0x21..=0x2f | 0x3a..=0x40 | 0x5b..=0x60 | 0x7b..=0x7e) &&
    offset + width < length &&
    input[offset + width] is (b'A'..=b'Z' | b'a'..=b'z') {
    for cursor = offset + width; cursor < length &&
       input[cursor] is (b'A'..=b'Z' | b'a'..=b'z'); {
      continue cursor + 1
    } nobreak {
      return cursor
    }
  }
  if ordered_word_scalar(first, script_fragment) {
    return ordered_word_end(input, length, offset, script_fragment)
  }
  if first != 0x0a &&
    first != 0x0d &&
    !qwen3_is_letter(first) &&
    !unicode_is_symbol(first) &&
    offset + width < length {
    let next = offset + width
    let (scalar, _) = qwen3_scalar_at(input, length, next)
    if ordered_word_scalar(scalar, script_fragment) {
      return ordered_word_end(input, length, next, script_fragment)
    }
  }
  let symbol_start = if first == 0x20 && offset + width < length {
    let (next, _) = qwen3_scalar_at(input, length, offset + width)
    if unicode_is_symbol(next) && ordered_script_range(next) == script_fragment {
      offset + width
    } else {
      offset
    }
  } else {
    offset
  }
  let (symbol, _) = qwen3_scalar_at(input, length, symbol_start)
  if unicode_is_symbol(symbol) {
    let end = for cursor = symbol_start; cursor < length; {
      let (scalar, w) = qwen3_scalar_at(input, length, cursor)
      if !unicode_is_symbol(scalar) ||
        ordered_script_range(scalar) != script_fragment {
        break cursor
      }
      continue cursor + w
    } nobreak {
      cursor
    }
    for cursor = end; !script_fragment &&
       cursor < length &&
       input[cursor] is (b'\r' | b'\n'); {
      continue cursor + 1
    } nobreak {
      return cursor
    }
  }
  if qwen3_is_whitespace(first) {
    for cursor = offset, previous = offset, newline = -1; cursor < length; {
      let (scalar, w) = qwen3_scalar_at(input, length, cursor)
      if !qwen3_is_whitespace(scalar) {
        return if newline >= 0 {
          newline
        } else if previous > offset {
          previous
        } else {
          cursor
        }
      }
      continue cursor + w,
        cursor,
        if scalar is (0x0a | 0x0d) {
          cursor + w
        } else {
          newline
        }
    } nobreak {
      return if newline >= 0 { newline } else { cursor }
    }
  }
  -1
}

///|
/// Streaming realization of the ordered split composition, with no heap allocation
/// or full-remainder rescan. Earlier isolated fragments cannot be rejoined by BPE.
fn ordered_script_pretoken_end(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
) -> Int raise TokenizerError {
  let (first, width) = qwen3_scalar_at(input, length, offset)
  if qwen3_is_number(first) {
    for cursor = offset + width, count = 1; cursor < length && count < 3; {
      let (scalar, w) = qwen3_scalar_at(input, length, cursor)
      if !qwen3_is_number(scalar) {
        return cursor
      }
      continue cursor + w, count + 1
    } nobreak {
      return cursor
    }
  }
  let script_fragment = ordered_script_range(first)
  let matched = ordered_word_match_end(input, length, offset, script_fragment)
  if matched >= 0 {
    return matched
  }
  for cursor = offset + width; cursor < length; {
    let (scalar, w) = qwen3_scalar_at(input, length, cursor)
    if qwen3_is_number(scalar) ||
      ordered_script_range(scalar) != script_fragment ||
      ordered_word_match_end(input, length, cursor, script_fragment) >= 0 {
      break cursor
    }
    continue cursor + w
  } nobreak {
    cursor
  }
}
