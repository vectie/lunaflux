///|
fn PrefixIndex::page_precedes(
  self : PrefixIndex,
  left : @page_allocator.PageId,
  right : @page_allocator.PageId,
) -> Bool {
  self.duplicate_page_comparisons += 1
  left.index() < right.index() ||
  (left.index() == right.index() && left.generation() < right.generation())
}

///|
fn PrefixIndex::duplicate_page_swap(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Unit {
  if left == right {
    return
  }
  let left_page = try! self.duplicate_page_scratch.read(left)
  let right_page = try! self.duplicate_page_scratch.read(right)
  try! self.duplicate_page_scratch.write(left, right_page)
  try! self.duplicate_page_scratch.write(right, left_page)
  self.duplicate_page_swaps += 1
}

///|
fn PrefixIndex::duplicate_page_sift_down(
  self : PrefixIndex,
  start : Int,
  length : Int,
) -> Unit {
  let mut parent = start
  while parent < length / 2 {
    let left = parent * 2 + 1
    let right = left + 1
    let mut greatest = left
    if right < length &&
      self.page_precedes(
        try! self.duplicate_page_scratch.read(left),
        try! self.duplicate_page_scratch.read(right),
      ) {
      greatest = right
    }
    if !self.page_precedes(
        try! self.duplicate_page_scratch.read(parent),
        try! self.duplicate_page_scratch.read(greatest),
      ) {
      break
    }
    self.duplicate_page_swap(parent, greatest)
    parent = greatest
  }
}

///|
fn PrefixIndex::clear_duplicate_page_scratch(
  self : PrefixIndex,
  length : Int,
) -> Unit {
  for index in 0..<length {
    try! self.duplicate_page_scratch.clear(index)
  }
}

///|
fn PrefixIndex::publish_pages_have_duplicate(
  self : PrefixIndex,
  pages : PrefixPublishBuffer,
) -> Bool {
  let length = pages.length()
  self.duplicate_page_comparisons = 0
  self.duplicate_page_swaps = 0
  for index in 0..<length {
    try! self.duplicate_page_scratch.write(index, try! pages.page_at(index))
  }
  let mut parent = length / 2 - 1
  while parent >= 0 {
    self.duplicate_page_sift_down(parent, length)
    parent -= 1
  }
  let mut end = length - 1
  while end > 0 {
    self.duplicate_page_swap(0, end)
    self.duplicate_page_sift_down(0, end)
    end -= 1
  }
  let mut duplicate = false
  for index in 1..<length {
    self.duplicate_page_comparisons += 1
    if (try! self.duplicate_page_scratch.read(index - 1)) ==
      (try! self.duplicate_page_scratch.read(index)) {
      duplicate = true
      break
    }
  }
  self.clear_duplicate_page_scratch(length)
  duplicate
}
