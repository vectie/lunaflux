///|
test "exact pool storage plan scales all backing and admits precisely its own bound" {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let framed = @framed_wire.FramedWireLimits::new(
    inference,
    max_frame_bytes=4096,
  )
  let one = LunaRequestPreparationStorageBudget::required(
    tokenizer, inference, framed, 1,
  )
  let many = LunaRequestPreparationStorageBudget::required(
    tokenizer, inference, framed, 32,
  )
  let (i, b, r) = one.cells()
  assert_eq(many.cells(), (i * 32UL, b * 32UL, r * 32UL))
  many.require_fits(tokenizer, inference, framed, 32)
  for
    short_budget in [
      LunaRequestPreparationStorageBudget::new(
        int_cells=i * 32UL - 1UL,
        byte_cells=b * 32UL,
        reference_cells=r * 32UL,
      ),
      LunaRequestPreparationStorageBudget::new(
        int_cells=i * 32UL,
        byte_cells=b * 32UL - 1UL,
        reference_cells=r * 32UL,
      ),
      LunaRequestPreparationStorageBudget::new(
        int_cells=i * 32UL,
        byte_cells=b * 32UL,
        reference_cells=r * 32UL - 1UL,
      ),
    ] {
    try short_budget.require_fits(tokenizer, inference, framed, 32) catch {
      Invalid(LunaPreparationStorage) => ()
      _ => fail("unexpected storage failure")
    } noraise {
      _ => fail("undersized plan admitted")
    }
  }
}
