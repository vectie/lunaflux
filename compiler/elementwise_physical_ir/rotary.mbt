///|
/// Split-half rotary work owns a frequency pair across all Q/K heads; the
/// remaining packed value suffix is a disjoint bit-exact copy domain.
pub struct RotaryProgram {
  priv pairs : Int
  priv rotary_heads : Int
  priv rotary_width : Int
  priv copy_groups : Int
  priv owners : Int
} derive(Eq, Debug)

///|
pub extend RotaryProgram with Eq::{equal, not_equal}

///|
pub extend RotaryProgram with @debug.Debug::{to_repr}

///|
pub fn RotaryProgram::new(
  width~ : Int,
  query_heads~ : Int,
  key_heads~ : Int,
  head_dimension~ : Int,
  owners~ : Int,
) -> RotaryProgram? {
  guard width > 0 &&
    query_heads > 0 &&
    key_heads > 0 &&
    head_dimension > 0 &&
    head_dimension % 2 == 0 &&
    owners > 0 else {
    return None
  }
  let heads = query_heads.to_int64() + key_heads.to_int64()
  let rotary = heads * head_dimension.to_int64()
  guard heads <= owners.to_int64() && rotary <= width.to_int64() else {
    return None
  }
  let suffix = width - rotary.to_int()
  Some({
    pairs: head_dimension / 2,
    rotary_heads: heads.to_int(),
    rotary_width: rotary.to_int(),
    copy_groups: if suffix == 0 {
      0
    } else {
      (suffix - 1) / owners + 1
    },
    owners,
  })
}

///|
pub fn RotaryProgram::geometry(
  self : RotaryProgram,
) -> (Int, Int, Int, Int, Int) {
  (
    self.pairs,
    self.rotary_heads,
    self.rotary_width,
    self.copy_groups,
    self.owners,
  )
}

///|
pub fn RotaryProgram::work_per_token(self : RotaryProgram) -> Int {
  self.pairs + self.copy_groups
}
