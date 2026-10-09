///|
pub(all) suberror RotaryError {
  InvalidFrequency
  ElementLimitExceeded
}

///|
pub struct RotaryTables {
  priv width : Int
  priv cosine : ReadOnlyArray[Float]
  priv sine : ReadOnlyArray[Float]
}

///|
/// Startup host coefficient preparation for the full-width F32 attention ABI.
/// Use checkpoint rope.inv_freq, never regenerate it from an assumed base.
/// Cast coordinates to F32 before multiplication; concatenate t/h/w and repeat
/// the half. Host transcendental results still require device qualification.
pub fn PositionedSequence::rotary_tables(
  self : Self,
  inv_freq : ArrayView[Float],
  max_elements~ : Int64,
) -> RotaryTables raise RotaryError {
  guard inv_freq.length() > 0 && inv_freq.length() <= 357913941 else {
    raise InvalidFrequency
  }
  for frequency in inv_freq {
    guard !frequency.is_nan() && !frequency.is_inf() && frequency > 0.0F else {
      raise InvalidFrequency
    }
  }
  let width = inv_freq.length() * 6
  let elements = self.positions.length().to_int64() * width.to_int64()
  guard elements <= 2147483647L &&
    max_elements > 0L &&
    elements <= max_elements / 2L else {
    raise ElementLimitExceeded
  }
  let cosine = Array::make(elements.to_int(), 0.0F)
  let sine = Array::make(elements.to_int(), 0.0F)
  let half = width / 2
  for row, position in self.positions {
    for axis, coordinate in [position.time, position.height, position.width] {
      let rounded = Float::from_double(coordinate)
      for lane, frequency in inv_freq {
        let phase = rounded * frequency
        guard !phase.is_nan() && !phase.is_inf() else { raise InvalidFrequency }
        let c = Float::from_double(@math.cos(phase.to_double()))
        let s = Float::from_double(@math.sin(phase.to_double()))
        let index = row * width + axis * inv_freq.length() + lane
        cosine[index] = c
        cosine[index + half] = c
        sine[index] = s
        sine[index + half] = s
      }
    }
  }
  {
    width,
    cosine: ReadOnlyArray::from_array(cosine),
    sine: ReadOnlyArray::from_array(sine),
  }
}

///|
pub fn RotaryTables::width(self : Self) -> Int {
  self.width
}

///|
pub fn RotaryTables::cosine(self : Self) -> ReadOnlyArray[Float] {
  self.cosine
}

///|
pub fn RotaryTables::sine(self : Self) -> ReadOnlyArray[Float] {
  self.sine
}
