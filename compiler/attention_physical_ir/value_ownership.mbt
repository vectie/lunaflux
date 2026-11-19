///|
/// Independent value components may be reassigned without changing the ordered
/// key fold. A contiguous fragment is legal only when owners cover the exact
/// dimension; ragged dimensions retain the interleaved masked map.
pub struct ValueComponentOwnership {
  priv dimensions : Int32
  priv owners : Int32
  priv items : Int32
  priv contiguous : Bool
} derive(Debug, Eq)

///|
pub fn ValueComponentOwnership::new(
  dimensions~ : Int32,
  owners~ : Int32,
  maximum_fragment~ : Int32,
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
) -> Int32 {
  if self.contiguous {
    self.items
  } else {
    1
  }
}

///|
pub fn ValueComponentOwnership::component(
  self : ValueComponentOwnership,
  owner : Int32,
  item : Int32,
) -> Int32? {
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
