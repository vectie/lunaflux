///|
fn valid_usage_scalars(
  inference : @inference.InferenceLimits,
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
) -> Bool {
  input_tokens >= 0 &&
  input_tokens <= inference.max_context_tokens() &&
  cached_input_tokens >= 0 &&
  cached_input_tokens <= input_tokens &&
  output_tokens >= 0 &&
  output_tokens <= inference.max_new_tokens() &&
  input_tokens <= inference.max_context_tokens() - output_tokens
}

///|
fn write_usage_scalars(
  bytes : FixedArray[Byte],
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
) -> Unit {
  write_i32(bytes, 32, input_tokens)
  write_i32(bytes, 36, cached_input_tokens)
  write_i32(bytes, 40, output_tokens)
  write_i32(bytes, 44, input_tokens + output_tokens)
}
