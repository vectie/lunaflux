///|
/// Native Int backing is four bytes and native reference backing eight on the
/// supported 64-bit service target. This excludes allocator/object overhead;
/// the external process/host memory envelope remains authoritative.
fn native_preparation_backing_bytes(
  cells : (UInt64, UInt64, UInt64),
  ceiling : UInt64,
) -> UInt64? {
  let (ints, bytes, references) = cells
  guard ints <= ceiling / 4UL && references <= ceiling / 8UL else {
    return None
  }
  let integer_bytes = ints * 4UL
  guard bytes <= ceiling - integer_bytes else { return None }
  let total = integer_bytes + bytes
  guard references * 8UL <= ceiling - total else { return None }
  Some(total + references * 8UL)
}

///|
fn ServicePlan::preparation_storage(
  self : Self,
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  framed : @framed.FramedWireLimits,
) -> @admission.LunaRequestPreparationStorageBudget raise {
  let storage = @admission.LunaRequestPreparationStorageBudget::required(
    tokenizer,
    inference,
    framed,
    self.slots * 2,
  )
  guard native_preparation_backing_bytes(
      storage.cells(),
      self.reserve.to_uint64(),
    )
    is Some(bytes) else {
    fail("HTTP preparation backing exceeds service reserve before allocation")
  }
  println(
    "preparation lanes=\{self.slots*2} backing_bytes=\{bytes} reserve_bytes=\{self.reserve}",
  )
  storage
}

///|
test "host preparation backing is independently bounded without arithmetic overflow" {
  assert_eq(native_preparation_backing_bytes((2UL, 3UL, 1UL), 19UL), Some(19UL))
  assert_eq(native_preparation_backing_bytes((2UL, 3UL, 1UL), 18UL), None)
  assert_eq(
    native_preparation_backing_bytes((0xffffffffffffffffUL, 0UL, 0UL), 19UL),
    None,
  )
}
