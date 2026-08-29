///|
pub fn RankGroupGeneration::from_wire(
  value : UInt64,
) -> RankGroupGeneration raise RankGroupError {
  if value == 0UL {
    raise Invalid(GroupGeneration)
  }
  { value, }
}

///|
pub fn RankGroupGeneration::first() -> RankGroupGeneration {
  { value: 1UL }
}

///|
pub fn RankGroupGeneration::value(self : RankGroupGeneration) -> UInt64 {
  self.value
}

///|
pub fn RankGroupGeneration::next(
  self : RankGroupGeneration,
) -> RankGroupGeneration raise RankGroupError {
  if self.value == 0xffffffffffffffffUL {
    raise Invalid(GroupGeneration)
  }
  { value: self.value + 1UL }
}
