///|
pub fn dsa_kv_attention_assembly(
  packed_kv : ArrayView[Double],
  shared_key_suffix : ArrayView[Double],
  row_count~ : Int,
  head_count~ : Int,
  key_nope_width~ : Int,
  shared_key_suffix_width~ : Int,
  value_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmDsaKvAssemblyResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(head_count)
  limits.check_channels(key_nope_width + shared_key_suffix_width)
  limits.check_channels(value_width)
  if head_count <= 0 ||
    key_nope_width <= 0 ||
    shared_key_suffix_width < 0 ||
    value_width <= 0 ||
    packed_kv.length() !=
    row_count * head_count * (key_nope_width + value_width) ||
    shared_key_suffix.length() != row_count * shared_key_suffix_width {
    raise InvalidShape
  }
  require_finite(packed_kv)
  require_finite(shared_key_suffix)
  let key_width = key_nope_width + shared_key_suffix_width
  let keys = Array::make(row_count * head_count * key_width, 0.0)
  let values = Array::make(row_count * head_count * value_width, 0.0)
  for row in 0..<row_count {
    for head in 0..<head_count {
      let head_row = row * head_count + head
      let packed_base = head_row * (key_nope_width + value_width)
      let key_base = head_row * key_width
      let value_base = head_row * value_width
      for component in 0..<key_nope_width {
        keys[key_base + component] = packed_kv[packed_base + component]
      }
      for component in 0..<shared_key_suffix_width {
        keys[key_base + key_nope_width + component] = shared_key_suffix[row *
          shared_key_suffix_width +
          component]
      }
      for component in 0..<value_width {
        values[value_base + component] = packed_kv[packed_base +
          key_nope_width +
          component]
      }
    }
  }
  {
    row_count,
    head_count,
    key_width,
    value_width,
    keys: ReadOnlyArray::from_array(keys),
    values: ReadOnlyArray::from_array(values),
  }
}

///|
pub fn GlmDsaKvAssemblyResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmDsaKvAssemblyResult::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmDsaKvAssemblyResult::key_width(self : Self) -> Int {
  self.key_width
}

///|
pub fn GlmDsaKvAssemblyResult::value_width(self : Self) -> Int {
  self.value_width
}

///|
pub fn GlmDsaKvAssemblyResult::keys(self : Self) -> ReadOnlyArray[Double] {
  self.keys
}

///|
pub fn GlmDsaKvAssemblyResult::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}
