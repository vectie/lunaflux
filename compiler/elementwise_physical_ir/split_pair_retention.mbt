///|
/// Register ownership of a split-half pair. Both components have the same
/// owner as the original strided component fold, so retaining them removes
/// scratch round trips without changing either reduction order or rounding.
pub struct SplitPairRetention {
  priv half : Int32
  priv owners : Int32
} derive(Eq, Debug)

///|
pub extend SplitPairRetention with Eq::{equal, not_equal}

///|
pub extend SplitPairRetention with @debug.Debug::{to_repr}

///|
pub fn SplitPairRetention::new(
  width~ : Int32,
  owners~ : Int32,
) -> SplitPairRetention? {
  guard width > 0 && width % 2 == 0 && owners > 0 && width / 2 % owners == 0 else {
    return None
  }
  Some({ half: width / 2, owners, })
}

///|
/// (split-half width, cooperative owners, retained pairs per owner).
pub fn SplitPairRetention::geometry(
  self : SplitPairRetention,
) -> (Int32, Int32, Int32) {
  (self.half, self.owners, self.half / self.owners)
}

///|
pub fn SplitPairRetention::components(
  self : SplitPairRetention,
  owner : Int32,
  pair : Int32,
) -> (Int32, Int32)? {
  guard owner >= 0 &&
    owner < self.owners &&
    pair >= 0 &&
    pair < self.half / self.owners else {
    return None
  }
  let first = owner + pair * self.owners
  Some((first, first + self.half))
}
