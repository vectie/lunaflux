///|
/// Lower the single-owner accumulator map to coalesced packed output stores.
/// Conversion is pointwise; register exchange changes ownership, never the
/// ordered fold or rounding operation. No shared result lifetime is created.
fn projection_register_exchange_source(output_width : Int) -> String {
  (
    $|      #pragma unroll
    $|      for (int store = 0; store < 4; ++store) {
    $|        const int half_row = (store / 2) * 2;
    $|        const __nv_bfloat162 low = __floats2bfloat162_rn(c[row][half_row], c[row][half_row + 1]);
    $|        const __nv_bfloat162 high = __floats2bfloat162_rn(c[row][half_row + 4], c[row][half_row + 5]);
    $|        const int source_lane = (lane / 8 + (store % 2) * 4) * 4 + lane % 4;
    $|        const unsigned low_bits = __shfl_sync(0xffffffffU, *reinterpret_cast<const unsigned *>(&low), source_lane);
    $|        const unsigned high_bits = __shfl_sync(0xffffffffU, *reinterpret_cast<const unsigned *>(&high), source_lane);
    $|        const unsigned bits = (lane & 4) ? high_bits : low_bits;
    $|        const int output_row = row_start + row * 16 + store * 4 + lane / 8;
    $|        if (output_row < token_count) *reinterpret_cast<unsigned *>(output + (unsigned long long)output_row * \{output_width}ULL + column + (lane % 8) * 2) = bits;
    $|      }
  )
}
