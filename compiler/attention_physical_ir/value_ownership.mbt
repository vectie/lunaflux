///|
/// Independent value components may be reassigned without changing the ordered
/// key fold. A contiguous fragment is legal only when owners cover the exact
/// dimension; ragged dimensions retain the interleaved masked map.
pub struct ValueComponentOwnership {
  priv dimensions : Int
  priv owners : Int
  priv items : Int
  priv contiguous : Bool
} derive(Debug, Eq)

///|
pub fn ValueComponentOwnership::new(
  dimensions~ : Int,
  owners~ : Int,
  maximum_fragment~ : Int,
) -> ValueComponentOwnership? {
  guard dimensions > 0 && owners > 0 && maximum_fragment > 0 else {
    return None
  }
  let items = (dimensions - 1) / owners + 1
  Some({
    dimensions,
    owners,
    items,
    contiguous: dimensions % owners == 0 && items <= maximum_fragment,
  })
}

///|
pub fn ValueComponentOwnership::fragment_width(
  self : ValueComponentOwnership,
) -> Int {
  if self.contiguous {
    self.items
  } else {
    1
  }
}

///|
pub fn ValueComponentOwnership::component(
  self : ValueComponentOwnership,
  owner : Int,
  item : Int,
) -> Int? {
  guard owner >= 0 && owner < self.owners && item >= 0 && item < self.items else {
    return None
  }
  let component = if self.contiguous {
    owner * self.items + item
  } else {
    owner + item * self.owners
  }
  if component < self.dimensions {
    Some(component)
  } else {
    None
  }
}
