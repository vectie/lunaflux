///|
/// Split-half rotary work owns a frequency pair across all Q/K heads; the
/// remaining packed value suffix is a disjoint bit-exact copy domain.
pub struct RotaryProgram {
  priv pairs : Int32
  priv rotary_heads : Int32
  priv rotary_width : Int32
  priv copy_groups : Int32
  priv owners : Int32
} derive(Eq, Debug)

///|
pub extend RotaryProgram with Eq::{equal, not_equal}

///|
pub extend RotaryProgram with @debug.Debug::{to_repr}

///|
pub fn RotaryProgram::new(
  width~ : Int32,
  query_heads~ : Int32,
  key_heads~ : Int32,
  head_dimension~ : Int32,
  owners~ : Int32,
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
) -> (Int32, Int32, Int32, Int32, Int32) {
  (
    self.pairs,
    self.rotary_heads,
    self.rotary_width,
    self.copy_groups,
    self.owners,
  )
}

///|
pub fn RotaryProgram::work_per_token(self : RotaryProgram) -> Int32 {
  self.pairs + self.copy_groups
}
