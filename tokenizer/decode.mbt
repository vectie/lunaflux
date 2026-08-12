///|
/// Decode token IDs back to their exact configured byte pieces.
pub fn TokenizerSpec::decode_bytes(
  self : TokenizerSpec,
  token_ids : ArrayView[Int],
  special_policy~ : DecodeSpecialPolicy,
) -> Bytes raise TokenizerError {
  if token_ids.length() > self.limits.max_output_tokens {
    raise OutputTooLong(
      actual=token_ids.length(),
      maximum=self.limits.max_output_tokens,
    )
  }
  let mut total = 0
  for token_id in token_ids {
    match self.tokens_by_id.get(token_id) {
      None => raise UnknownTokenId(token_id~)
      Some(token) =>
        if !(token.kind is Special && special_policy is Skip) {
          if token.piece.length() > self.limits.max_decoded_bytes - total {
            raise DecodedOutputTooLong(maximum=self.limits.max_decoded_bytes)
          }
          total += token.piece.length()
        }
    }
  }
  let output : Array[Byte] = Array::new(capacity=total)
  for token_id in token_ids {
    let token = self.tokens_by_id[token_id]
    if !(token.kind is Special && special_policy is Skip) {
      output.push_iter(token.piece.iter())
    }
  }
  Bytes::from_array(output)
}

///|
/// Decode token IDs as strict UTF-8 text.
///
/// Use `decode_bytes` when arbitrary byte output is valid for the caller.
pub fn TokenizerSpec::decode_text(
  self : TokenizerSpec,
  token_ids : ArrayView[Int],
  special_policy~ : DecodeSpecialPolicy,
) -> String raise TokenizerError {
  let bytes = self.decode_bytes(token_ids, special_policy~)
  @utf8.decode(bytes) catch {
    _ => raise InvalidUtf8Output
  }
}
