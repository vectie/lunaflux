///|
pub(all) suberror Qwen3TokenBridgeError {
  MalformedRequest
  UnsupportedRequest
  RequestTooLarge
} derive(Eq, Debug)

///|
pub(all) suberror Qwen3DetokenizerError {
  InvalidDetokenizer
  DetokenizerFinished
  TokenDecodeFailed
} derive(Eq, Debug)

///|
pub struct Qwen3TokenIdRequest {
  priv input_token_ids : Array[Int]
  priv max_output_tokens : Int
}

///|
pub fn Qwen3TokenIdRequest::input_token_ids(
  self : Qwen3TokenIdRequest,
) -> ArrayView[Int] {
  self.input_token_ids
}

///|
pub fn Qwen3TokenIdRequest::max_output_tokens(
  self : Qwen3TokenIdRequest,
) -> Int {
  self.max_output_tokens
}

///|
pub struct Qwen3TokenSseBuffer {
  priv storage : FixedArray[Byte]
  priv digits : FixedArray[Byte]
}

///|
pub fn Qwen3TokenSseBuffer::new(
  max_text_bytes? : Int = 4096,
) -> Qwen3TokenSseBuffer {
  {
    storage: FixedArray::make(160 + Int::max(max_text_bytes, 0) * 6, b'\x00'),
    digits: FixedArray::make(10, b'0'),
  }
}
