///|
fn kvp_allocator() -> @page_allocator.PageAllocator {
  @page_allocator.PageAllocator::new(
    try! @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=4,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
      max_generation=8,
    ),
  )
}

///|
test "every rank preserves one logical page identity and deterministic slot" {
  let plan = kvp_admit()
  let allocator = kvp_allocator()
  let page = allocator.allocate_one()
  let generation = kvp_generation()
  let rank_zero = plan.page_region(generation, 0, page, 1, Key)
  let rank_one = plan.page_region(generation, 1, page, 1, Key)
  assert_eq(rank_zero.page_index(), page.index())
  assert_eq(rank_one.page_index(), page.index())
  assert_eq(rank_zero.page_index(), rank_one.page_index())
  assert_eq(rank_zero.page_generation(), page.generation())
  assert_eq(rank_one.page_generation(), page.generation())
  assert_eq(rank_zero.offset_bytes(), rank_one.offset_bytes())
  assert_eq(rank_zero.logical_bytes(), rank_one.logical_bytes())
  assert_eq(rank_zero.reserved_bytes(), rank_one.reserved_bytes())
  assert_eq(
    plan.element_offset_bytes(generation, 0, page, 1, Value, 15, 3, 63),
    plan.element_offset_bytes(generation, 1, page, 1, Value, 15, 3, 63),
  )
}

///|
test "page and element projections fail closed on every substituted index" {
  let plan = kvp_admit()
  let allocator = kvp_allocator()
  ignore(allocator.allocate_one())
  ignore(allocator.allocate_one())
  ignore(allocator.allocate_one())
  ignore(allocator.allocate_one())
  let small_plan = kvp_admit(pages=2)
  let page = allocator.allocate_one() catch {
    _ => {
      let replacement = kvp_allocator()
      ignore(replacement.allocate_one())
      ignore(replacement.allocate_one())
      replacement.allocate_one()
    }
  }
  let generation = kvp_generation()
  kvp_expect(
    () => ignore(small_plan.page_region(generation, 0, page, 0, Key)),
    PageIndex,
  )
  let valid_allocator = kvp_allocator()
  let valid = valid_allocator.allocate_one()
  kvp_expect(() => ignore(plan.page_region(generation, 2, valid, 0, Key)), Rank)
  kvp_expect(
    () => ignore(plan.page_region(generation, 0, valid, 2, Key)),
    LayerIndex,
  )
  kvp_expect(
    () => {
      ignore(plan.element_offset_bytes(generation, 0, valid, 0, Key, 16, 0, 0))
    },
    TokenIndex,
  )
  kvp_expect(
    () => {
      ignore(plan.element_offset_bytes(generation, 0, valid, 0, Key, 0, 4, 0))
    },
    LocalKvHeadIndex,
  )
  kvp_expect(
    () => {
      ignore(plan.element_offset_bytes(generation, 0, valid, 0, Key, 0, 0, 64))
    },
    HeadDimensionIndex,
  )
}
