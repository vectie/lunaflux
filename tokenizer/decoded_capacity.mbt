///|
/// Pure startup geometry over admitted, decoded vocabulary pieces. Includes
/// specials, so either emission policy fits. Decoder initial-space stripping
/// can only reduce this bound. No token-step scan or output allocation.
pub fn TokenizerSpec::maximum_decoded_piece_bytes(self : Self) -> Int32 {
  let mut maximum = 0
  for _, token in self.tokens_by_id {
    maximum = Int32::max(maximum, token.decoded_piece.length())
  }
  maximum
}
