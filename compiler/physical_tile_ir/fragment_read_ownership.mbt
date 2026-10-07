///|
/// A bijective relabeling of packed read-provider groups. Each group supplies
/// one packed word; consumers receive contiguous words in their original
/// semantic order. This changes read ownership, not shared storage or values.
pub struct FragmentReadOwnership {
  priv forwarding : FragmentForwarding
  priv group_width : Int
} derive(Eq, Debug)

///|
pub extend FragmentReadOwnership with Eq::{equal, not_equal}

///|
pub extend FragmentReadOwnership with @debug.Debug::{to_repr}

///|
pub fn FragmentForwarding::contiguous_read(
  self : FragmentForwarding,
  provider_group_width~ : Int,
) -> FragmentReadOwnership? {
  guard provider_group_width > 0 &&
    provider_group_width.to_int64() * self.geometry().0.to_int64() <=
    2147483647L else {
    return None
  }
  Some({ forwarding: self, group_width: provider_group_width, })
}

///|
pub fn FragmentReadOwnership::provider(
  self : FragmentReadOwnership,
  owner : Int,
) -> Int? {
  let (words, per_instruction, _) = self.forwarding.geometry()
  guard owner >= 0 && owner < words * self.group_width else { return None }
  let word = owner / self.group_width
  let source = self.forwarding
    .word(word / per_instruction, word % per_instruction)
    .unwrap()
  Some(source * self.group_width + owner % self.group_width)
}

///|
pub fn FragmentReadOwnership::source_word(
  self : FragmentReadOwnership,
  word : Int,
) -> Int? {
  let (_, per_instruction, _) = self.forwarding.geometry()
  self.forwarding.word(word / per_instruction, word % per_instruction)
}

///|
pub fn FragmentReadOwnership::canonical(self : FragmentReadOwnership) -> String {
  "fragment-contiguous-read-v1;group-width=\{self.group_width};source=\{self.forwarding.canonical()}"
}
