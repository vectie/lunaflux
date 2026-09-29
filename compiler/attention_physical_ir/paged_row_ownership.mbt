///|
/// Address ownership for a contiguous row split into equal vector fragments.
/// Sharing is legal only when each owner group is contained in one subgroup
/// and every subgroup participant reaches the distribution operation.
pub struct PagedRowOwnership {
  priv consumers : Int
  priv shared : Bool
} derive(Debug, Eq)

///|
pub extend PagedRowOwnership with Eq::{equal, not_equal}

///|
pub extend PagedRowOwnership with @debug.Debug::{to_repr}

///|
pub fn PagedRowOwnership::new(
  row_elements~ : Int,
  vector_elements~ : Int,
  subgroup_size~ : Int,
  hoist~ : Bool,
) -> PagedRowOwnership? {
  if row_elements <= 0 ||
    vector_elements <= 0 ||
    subgroup_size <= 0 ||
    row_elements % vector_elements != 0 {
    return None
  }
  let consumers = row_elements / vector_elements
  Some({
    consumers,
    shared: hoist &&
    consumers <= subgroup_size &&
    subgroup_size % consumers == 0,
  })
}

///|
pub fn PagedRowOwnership::shares_address(self : PagedRowOwnership) -> Bool {
  self.shared
}

///|
pub fn PagedRowOwnership::consumers(self : PagedRowOwnership) -> Int {
  self.consumers
}

///|
pub fn PagedRowOwnership::address_owner(
  self : PagedRowOwnership,
  owner : Int,
) -> Int {
  if self.shared {
    owner / self.consumers * self.consumers
  } else {
    owner
  }
}
