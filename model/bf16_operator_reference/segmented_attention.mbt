///|
/// Offline reference for staged BF16 segmented attention. Scratch is retained
/// between calls; host exp/sum order is not a GPU/library bitwise oracle.
pub struct SegmentedAttention {
  priv plan : @precision.SegmentedAttentionPrecision
  priv scores : FixedArray[Float]
}

///|
pub fn SegmentedAttention::new(
  plan : @precision.SegmentedAttentionPrecision,
) -> Self {
  let (_, segment, _, _) = plan.geometry()
  { plan, scores: FixedArray::make(segment, 0.0F), }
}

///|
pub fn SegmentedAttention::run_into(
  self : Self,
  starts : ArrayView[Int32],
  ends : ArrayView[Int32],
  query : ArrayView[Float],
  key : ArrayView[Float],
  value : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (capacity, maximum_segment, heads, width) = self.plan.geometry()
  let elements = rows.to_int64() * heads.to_int64() * width.to_int64()
  guard rows > 0 &&
    rows <= capacity &&
    starts.length() == rows &&
    ends.length() == rows &&
    query.length().to_int64() == elements &&
    key.length() == query.length() &&
    value.length() == query.length() &&
    output.length() == query.length() else {
    fail("invalid segmented attention buffers")
  }
  let scale = Float::from_double(1.0 / width.to_double().sqrt())
  for row in 0..<rows {
    let start = starts[row]
    let end = ends[row]
    guard start >= 0 &&
      start <= row &&
      end > row &&
      end <= rows &&
      end - start <= maximum_segment else {
      fail("invalid attention row span")
    }
    for head in 0..<heads {
      let qbase = (row * heads + head) * width
      let mut maximum = Float::reinterpret_from_uint(0xff800000U)
      for source in start..<end {
        let kbase = (source * heads + head) * width
        let mut sum = 0.0F
        for lane in 0..<width {
          sum += round_bf16(query[qbase + lane]) * round_bf16(key[kbase + lane])
        }
        let score = round_bf16(round_bf16(sum) * scale)
        self.scores[source - start] = score
        maximum = maximum.max(score)
      }
      let mut denominator = 0.0F
      for source in start..<end {
        let p = exp32(self.scores[source - start] - maximum)
        self.scores[source - start] = p
        denominator += p
      }
      for source in start..<end {
        self.scores[source - start] = round_bf16(
          self.scores[source - start] / denominator,
        )
      }
      for lane in 0..<width {
        let mut sum = 0.0F
        for source in start..<end {
          sum += self.scores[source - start] *
            round_bf16(value[(source * heads + head) * width + lane])
        }
        output[qbase + lane] = round_bf16(sum)
      }
    }
  }
}
