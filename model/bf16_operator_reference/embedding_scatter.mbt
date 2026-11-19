///|
/// Offline raw-word law. Text rows and decode preserve existing residual bits.
pub fn embedding_scatter_words_into(
  plan : @effects.EmbeddingScatterProgram,
  positions : ArrayView[Int32],
  feature_map : ArrayView[Int32],
  features : ArrayView[UInt32],
  residual : FixedArray[UInt32],
  rows~ : Int32,
  feature_rows~ : Int32,
  prefill~ : Bool,
) -> Unit raise {
  let (width, maximum_rows, maximum_positions, maximum_features) = plan.geometry()
  guard rows > 0 &&
    rows <= maximum_rows &&
    positions.length() == rows &&
    feature_map.length() <= maximum_positions &&
    feature_rows > 0 &&
    feature_rows <= maximum_features &&
    features.length().to_int64() == feature_rows.to_int64() * width.to_int64() &&
    residual.length().to_int64() >= rows.to_int64() * width.to_int64() else {
    fail("invalid embedding scatter buffers")
  }
  if !prefill {
    return
  }
  for row in 0..<rows {
    let position = positions[row]
    guard position >= 0 && position < feature_map.length() else {
      fail("invalid causal scatter position")
    }
    let feature = feature_map[position]
    guard feature >= -1 && feature < feature_rows else {
      fail("invalid scatter feature index")
    }
    if feature >= 0 {
      for lane in 0..<width {
        residual[row * width + lane] = features[feature * width + lane]
      }
    }
  }
}
