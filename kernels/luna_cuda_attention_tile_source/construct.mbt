///|
pub fn CudaPagedAttentionTileAbi::new(
  query_heads~ : Int,
  key_value_heads~ : Int,
  head_dimension~ : Int,
  tokens_per_page~ : Int,
  input_row_width~ : Int,
  page_stride_values~ : Int64,
  physical_page_capacity~ : Int64,
  maximum_query_rows~ : Int,
  maximum_query_tokens~ : Int,
  maximum_page_entries~ : Int,
) -> CudaPagedAttentionTileAbi raise CudaAttentionTileSourceError {
  if query_heads <= 0 ||
    key_value_heads <= 0 ||
    query_heads % key_value_heads != 0 ||
    !(head_dimension == 64 || head_dimension == 128) ||
    tokens_per_page <= 0 ||
    (tokens_per_page & (tokens_per_page - 1)) != 0 ||
    input_row_width < (query_heads + 2 * key_value_heads) * head_dimension ||
    page_stride_values <= 0L ||
    physical_page_capacity <= 0L ||
    maximum_query_rows <= 0 ||
    maximum_query_tokens < maximum_query_rows ||
    maximum_page_entries <= 0 {
    raise InvalidCudaAttentionTileSource(InvalidAbi)
  }
  {
    query_heads,
    key_value_heads,
    head_dimension,
    tokens_per_page,
    input_row_width,
    page_stride_values,
    physical_page_capacity,
    maximum_query_rows,
    maximum_query_tokens,
    maximum_page_entries,
  }
}

///|
fn valid_cuda_symbol(symbol : StringView) -> Bool {
  if symbol.length() == 0 || symbol.length() > 128 {
    return false
  }
  for index, unit in symbol {
    if !(unit == '_' ||
      unit is ('a'..='z' | 'A'..='Z') ||
      (index > 0 && unit is ('0'..='9'))) {
      return false
    }
  }
  true
}
