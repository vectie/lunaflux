///|
pub(all) enum CompressedRowReadSet {
  Learned(top_k~ : Int)
  AllCausal
}

///|
pub struct WindowCompressedKvPrecision {
  priv window : WindowSharedKvPrecision
  priv ratio : Int
  priv read_set : CompressedRowReadSet
  priv coordinates : CompressedRowCoordinates
  priv capacity : Int
  priv selected : Int
}

///|
pub fn WindowCompressedKvPrecision::new(
  window : WindowSharedKvPrecision,
  compression_ratio~ : Int,
  read_set~ : CompressedRowReadSet,
  coordinates? : CompressedRowCoordinates = OffsetRelative,
) -> Self raise PrecisionPlanError {
  guard compression_ratio > 1 && compression_ratio <= 1048576 else {
    raise IllegalPlacement
  }
  let (_, _, _, _, positions) = window.geometry()
  let capacity = positions / compression_ratio
  let selected = match read_set {
    Learned(top_k~) => {
      guard top_k > 0 && top_k <= 65536 else { raise IllegalPlacement }
      Int::min(top_k, capacity)
    }
    AllCausal => capacity
  }
  {
    window,
    ratio: compression_ratio,
    read_set,
    coordinates,
    capacity,
    selected,
  }
}

///|
pub fn WindowCompressedKvPrecision::coordinates(
  self : Self,
) -> CompressedRowCoordinates {
  self.coordinates
}

///|
pub fn WindowCompressedKvPrecision::window(
  self : Self,
) -> WindowSharedKvPrecision {
  self.window
}

///|
pub fn WindowCompressedKvPrecision::compression_ratio(self : Self) -> Int {
  self.ratio
}

///|
pub fn WindowCompressedKvPrecision::read_set(
  self : Self,
) -> CompressedRowReadSet {
  self.read_set
}

///|
pub fn WindowCompressedKvPrecision::cache_capacity(self : Self) -> Int {
  self.capacity
}

///|
pub fn WindowCompressedKvPrecision::selected_width(self : Self) -> Int {
  self.selected
}

///|
pub fn WindowCompressedKvPrecision::score_scale(self : Self) -> Double {
  let (_, width, _, _, _) = self.window.geometry()
  1.0 / width.to_double().sqrt()
}

///|
pub fn WindowCompressedKvPrecision::state_bytes(self : Self) -> Int64 {
  self.window.state_bytes()
}

///|
pub fn WindowCompressedKvPrecision::effect(_self : Self) -> NumericEffect {
  CacheCommit
}
