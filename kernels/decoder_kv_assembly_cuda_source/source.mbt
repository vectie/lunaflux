///|
pub(all) suberror DecoderKvAssemblyCudaSourceError {
  InvalidHeadCount
  InvalidKeyWidth
  InvalidValueWidth
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
pub(all) enum DecoderKvAssemblyNumericalContract {
  Bf16BitExactSplitAndSharedSuffixBroadcastV1
} derive(Debug, Eq)

///|
pub struct DecoderKvAssemblyCudaSource {
  priv head_count : Int
  priv key_nope_width : Int
  priv shared_key_suffix_width : Int
  priv value_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
pub fn decoder_kv_assembly_bf16_source(
  head_count~ : Int,
  key_nope_width~ : Int,
  shared_key_suffix_width~ : Int,
  value_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderKvAssemblyCudaSource raise DecoderKvAssemblyCudaSourceError {
  if head_count <= 0 || head_count > 1024 {
    raise InvalidHeadCount
  }
  if key_nope_width <= 0 ||
    key_nope_width > 32768 ||
    shared_key_suffix_width < 0 ||
    shared_key_suffix_width > 32768 {
    raise InvalidKeyWidth
  }
  if value_width <= 0 || value_width > 32768 {
    raise InvalidValueWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  let per_row = head_count.to_int64() *
    (key_nope_width + shared_key_suffix_width + value_width).to_int64()
  if per_row <= 0L || maximum_rows.to_int64() > 9223372036854775552L / per_row {
    raise DimensionOverflow
  }
  if function_symbol.is_empty() ||
    function_symbol.length() > 128 ||
    !(function_symbol[0] is ('a'..='z' | 'A'..='Z' | '_')) ||
    !function_symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
    raise InvalidFunctionSymbol
  }
  let signature = if shared_key_suffix_width > 0 {
    "(const int32_t *counts, const __nv_bfloat16 *packed_kv, const __nv_bfloat16 *shared_key_suffix, __nv_bfloat16 *keys, __nv_bfloat16 *values)"
  } else {
    "(const int32_t *counts, const __nv_bfloat16 *packed_kv, __nv_bfloat16 *keys, __nv_bfloat16 *values)"
  }
  let suffix_alias = if shared_key_suffix_width > 0 {
    " || keys == shared_key_suffix || values == shared_key_suffix"
  } else {
    ""
  }
  let suffix_copy = if shared_key_suffix_width > 0 {
    " else { keys[index] = shared_key_suffix[row * " +
    shared_key_suffix_width.to_string() +
    "ULL + component - " +
    key_nope_width.to_string() +
    "ULL]; }"
  } else {
    ""
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    function_symbol +
    signature +
    " {\n" +
    "  if (keys == values || keys == packed_kv || values == packed_kv" +
    suffix_alias +
    ") return; const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long key_width = " +
    (key_nope_width + shared_key_suffix_width).to_string() +
    "ULL; const unsigned long long key_cells = (unsigned long long)live_rows * " +
    head_count.to_string() +
    "ULL * key_width; const unsigned long long value_cells = (unsigned long long)live_rows * " +
    head_count.to_string() +
    "ULL * " +
    value_width.to_string() +
    "ULL; if (index >= key_cells + value_cells) return;\n" +
    "  if (index < key_cells) { const unsigned long long component = index % key_width; const unsigned long long head_row = index / key_width; const unsigned long long row = head_row / " +
    head_count.to_string() +
    "ULL; if (component < " +
    key_nope_width.to_string() +
    "ULL) { keys[index] = packed_kv[head_row * " +
    (key_nope_width + value_width).to_string() +
    "ULL + component]; }" +
    suffix_copy +
    " } else { const unsigned long long value_index = index - key_cells; const unsigned long long component = value_index % " +
    value_width.to_string() +
    "ULL; const unsigned long long head_row = value_index / " +
    value_width.to_string() +
    "ULL; values[value_index] = packed_kv[head_row * " +
    (key_nope_width + value_width).to_string() +
    "ULL + " +
    key_nope_width.to_string() +
    "ULL + component]; }\n}\n",
  )
  {
    head_count,
    key_nope_width,
    shared_key_suffix_width,
    value_width,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
pub fn DecoderKvAssemblyCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn DecoderKvAssemblyCudaSource::key_nope_width(self : Self) -> Int {
  self.key_nope_width
}

///|
pub fn DecoderKvAssemblyCudaSource::shared_key_suffix_width(self : Self) -> Int {
  self.shared_key_suffix_width
}

///|
pub fn DecoderKvAssemblyCudaSource::value_width(self : Self) -> Int {
  self.value_width
}

///|
pub fn DecoderKvAssemblyCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderKvAssemblyCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderKvAssemblyCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn DecoderKvAssemblyCudaSource::numerical_contract(
  _self : Self,
) -> DecoderKvAssemblyNumericalContract {
  Bf16BitExactSplitAndSharedSuffixBroadcastV1
}
