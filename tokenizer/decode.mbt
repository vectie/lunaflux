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

///|
/// Return the exact byte count contributed by one token under the selected
/// special-token policy. This performs no output allocation.
pub fn TokenizerSpec::decoded_piece_bytes(
  self : TokenizerSpec,
  token_id : Int,
  special_policy~ : DecodeSpecialPolicy,
) -> Int raise TokenizerError {
  guard self.tokens_by_id.get(token_id) is Some(token) else {
    raise UnknownTokenId(token_id~)
  }
  if token.kind is Special && special_policy is Skip {
    0
  } else {
    token.piece.length()
  }
}

///|
/// Copy one token's exact configured byte piece into caller-owned storage.
/// Capacity is proven before the first write, so rejection leaves the
/// destination unchanged. The returned count is zero for a skipped special
/// token and otherwise equals `decoded_piece_bytes` for the same arguments.
pub fn TokenizerSpec::copy_decoded_piece(
  self : TokenizerSpec,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  special_policy~ : DecodeSpecialPolicy,
) -> Int raise TokenizerError {
  guard self.tokens_by_id.get(token_id) is Some(token) else {
    raise UnknownTokenId(token_id~)
  }
  if destination_offset < 0 || destination_offset > destination.length() {
    raise InvalidDecodeDestination
  }
  let count = if token.kind is Special && special_policy is Skip {
    0
  } else {
    token.piece.length()
  }
  let available = destination.length() - destination_offset
  if count > available {
    raise DecodeDestinationTooSmall(required=count, available~)
  }
  for index in 0..<count {
    destination[destination_offset + index] = token.piece[index]
  }
  count
}

///|
/// Copies one decoded piece without constructing a typed error. Failure is
/// reported as an opaque scalar status and leaves the destination unchanged.
pub fn TokenizerSpec::copy_decoded_piece_status(
  self : TokenizerSpec,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  special_policy~ : DecodeSpecialPolicy,
) -> DecodePieceStatus {
  guard self.tokens_by_id.get(token_id) is Some(token) else {
    return DecodePieceStatus(1, 0)
  }
  if destination_offset < 0 || destination_offset > destination.length() {
    return DecodePieceStatus(2, 0)
  }
  let count = if token.kind is Special && special_policy is Skip {
    0
  } else {
    token.piece.length()
  }
  if count > destination.length() - destination_offset {
    return DecodePieceStatus(3, 0)
  }
  for index in 0..<count {
    destination[destination_offset + index] = token.piece[index]
  }
  DecodePieceStatus(0, count)
}
