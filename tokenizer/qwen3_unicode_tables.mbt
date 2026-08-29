///|
/// Immutable Unicode 17.0.0 data used only by the Qwen3 request tokenizer.
/// Source digests:
/// - UnicodeData.txt: 2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c
/// - DerivedNormalizationProps.txt:
///   71fd6a206a2c0cdd41feb6b7f656aa31091db45e9cedc926985d718397f9e488
/// Generated payload digests (concatenating both decomposition payloads):
/// - Letter ranges: ba05ad9eb8188d1c0c57ebba136aaa69f9ee2200a7142601e2668ec4eb7a56fd
/// - Combining classes: a6e8b7aabad1b83a9f8c5d5df07f723999b509c0b736e9c977e3df56bc94dcec
/// - Canonical decomposition:
///   bf87911fc2a804a126be4e61131b3866170893cec1880385b00ab23215406c0c
/// - Canonical composition:
///   08b3f5fa94fbf2879344f5ab74b490f7f06798f8fee654389da71b73a8b05273
/// The generated table declarations repeat the record counts. White-box
/// tests pin the exact byte lengths and representative boundary lookups.

///|
fn qwen3_unicode_u16(table : Bytes, offset : Int) -> Int {
  (table[offset].to_int() << 8) | table[offset + 1].to_int()
}

///|
fn qwen3_unicode_u24(table : Bytes, offset : Int) -> Int {
  (table[offset].to_int() << 16) |
  (table[offset + 1].to_int() << 8) |
  table[offset + 2].to_int()
}

///|
fn qwen3_is_letter(scalar : Int) -> Bool {
  let record_bytes = 6
  let mut low = 0
  let mut high = qwen3_unicode_letter_ranges.length() / record_bytes
  while low < high {
    let middle = low + (high - low) / 2
    let offset = middle * record_bytes
    let first = qwen3_unicode_u24(qwen3_unicode_letter_ranges, offset)
    let last = qwen3_unicode_u24(qwen3_unicode_letter_ranges, offset + 3)
    if scalar < first {
      high = middle
    } else if scalar > last {
      low = middle + 1
    } else {
      return true
    }
  }
  false
}

///|
fn qwen3_canonical_combining_class(scalar : Int) -> Int {
  let record_bytes = 7
  let mut low = 0
  let mut high = qwen3_unicode_combining_ranges.length() / record_bytes
  while low < high {
    let middle = low + (high - low) / 2
    let offset = middle * record_bytes
    let first = qwen3_unicode_u24(qwen3_unicode_combining_ranges, offset)
    let last = qwen3_unicode_u24(qwen3_unicode_combining_ranges, offset + 3)
    if scalar < first {
      high = middle
    } else if scalar > last {
      low = middle + 1
    } else {
      return qwen3_unicode_combining_ranges[offset + 6].to_int()
    }
  }
  0
}

///|
fn qwen3_canonical_decomposition(scalar : Int) -> (Int, Int)? {
  let record_bytes = 6
  let mut low = 0
  let mut high = qwen3_unicode_decomposition_index.length() / record_bytes
  while low < high {
    let middle = low + (high - low) / 2
    let offset = middle * record_bytes
    let candidate = qwen3_unicode_u24(qwen3_unicode_decomposition_index, offset)
    if scalar < candidate {
      high = middle
    } else if scalar > candidate {
      low = middle + 1
    } else {
      return Some(
        (
          qwen3_unicode_u16(qwen3_unicode_decomposition_index, offset + 3),
          qwen3_unicode_decomposition_index[offset + 5].to_int(),
        ),
      )
    }
  }
  None
}

///|
fn qwen3_decomposition_scalar(index : Int) -> Int {
  qwen3_unicode_u24(qwen3_unicode_decomposition_scalars, index * 3)
}

///|
fn qwen3_table_composition(first : Int, second : Int) -> Int? {
  let record_bytes = 9
  let mut low = 0
  let mut high = qwen3_unicode_composition_pairs.length() / record_bytes
  while low < high {
    let middle = low + (high - low) / 2
    let offset = middle * record_bytes
    let candidate_first = qwen3_unicode_u24(
      qwen3_unicode_composition_pairs, offset,
    )
    let candidate_second = qwen3_unicode_u24(
      qwen3_unicode_composition_pairs,
      offset + 3,
    )
    if first < candidate_first ||
      (first == candidate_first && second < candidate_second) {
      high = middle
    } else if first > candidate_first ||
      (first == candidate_first && second > candidate_second) {
      low = middle + 1
    } else {
      return Some(
        qwen3_unicode_u24(qwen3_unicode_composition_pairs, offset + 6),
      )
    }
  }
  None
}

///|
fn qwen3_canonical_composition(first : Int, second : Int) -> Int? {
  let l_base = 0x1100
  let v_base = 0x1161
  let t_base = 0x11a7
  let s_base = 0xac00
  let l_count = 19
  let v_count = 21
  let t_count = 28
  let n_count = v_count * t_count
  let s_count = l_count * n_count
  if first >= l_base &&
    first < l_base + l_count &&
    second >= v_base &&
    second < v_base + v_count {
    return Some(
      s_base + (first - l_base) * n_count + (second - v_base) * t_count,
    )
  }
  if first >= s_base &&
    first < s_base + s_count &&
    (first - s_base) % t_count == 0 &&
    second > t_base &&
    second < t_base + t_count {
    return Some(first + second - t_base)
  }
  qwen3_table_composition(first, second)
}

///|
let qwen3_nonzero_combining_classes : ReadOnlyArray[Int] = [
  1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 84, 91, 103, 107, 118, 122, 129, 130,
  132, 202, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 233, 234, 240,
]
