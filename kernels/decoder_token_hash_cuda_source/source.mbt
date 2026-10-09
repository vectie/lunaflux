///|
fn valid_token_hash_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Emits a row-major I32 table lookup. Runtime `counts[3]` selects the live
/// row prefix and is bounded by `maximum_rows`. Invalid token IDs or table
/// entries write the fail-closed sentinel `-1` without reading out of bounds.
/// Launch geometry is `[maximum_rows,1,1]` blocks of 256 threads.
pub fn token_hash_lookup_i32_source(
  vocabulary_size~ : Int,
  experts_per_token~ : Int,
  routed_expert_count~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderTokenHashCudaSource raise DecoderTokenHashCudaSourceError {
  if vocabulary_size <= 0 {
    raise InvalidVocabularySize
  }
  if experts_per_token <= 0 || experts_per_token > 256 {
    raise InvalidExpertsPerToken
  }
  if routed_expert_count <= 0 || experts_per_token > routed_expert_count {
    raise InvalidRoutedExpertCount
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_token_hash_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *token_ids, const int32_t *table, int32_t *expert_indices) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  const int32_t slot = (int32_t)threadIdx.x;\n" +
    "  if (row >= live_rows || slot >= " +
    experts_per_token.to_string() +
    ") return;\n" +
    "  const unsigned long long output_index = (unsigned long long)row * " +
    experts_per_token.to_string() +
    "ULL + (unsigned long long)slot;\n" +
    "  const int32_t token = token_ids[row];\n" +
    "  if (token < 0 || token >= " +
    vocabulary_size.to_string() +
    ") { expert_indices[output_index] = -1; return; }\n" +
    "  const unsigned long long table_index = (unsigned long long)token * " +
    experts_per_token.to_string() +
    "ULL + (unsigned long long)slot;\n" +
    "  const int32_t expert = table[table_index];\n" +
    "  expert_indices[output_index] = (expert >= 0 && expert < " +
    routed_expert_count.to_string() +
    ") ? expert : -1;\n" +
    "}\n",
  )
  {
    operation: TokenHashLookupI32,
    numerical_contract: BitExactInt32LookupV1,
    vocabulary_size,
    experts_per_token,
    routed_expert_count,
    maximum_rows,
    function_symbol,
    source,
  }
}
