///|
fn denoiser_test_packings() -> Array[Packing] raise {
  let metadata = @h3.MiniMaxH3ModelMetadata::from_verified_content(
    @h3.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    @spec.ContentDigest::from_sha256(String::make(64, '8')),
  )
  denoiser_packings(@minimax_h3.build(metadata), max_bytes=103079215104L)
}

///|
test "typed denoiser join preserves real layer ownership under reordered uploads" {
  let packings = denoiser_test_packings()
  let entries = Array::makei(packings.length(), i => {
    let index = packings.length() - 1 - i
    (packings[index], index)
  })
  let bound = denoiser_bindings(entries[:], 50, 2)
  assert_eq(bound.input.1, 0)
  assert_eq(bound.timestep.1, 1)
  assert_eq(bound.output.1, 2)
  assert_eq(bound.context.1, 3)
  for i, (packing, allocation) in bound.refiners {
    assert_eq(packing.layer, i)
    assert_eq(allocation, i + 4)
  }
  for i, (packing, allocation) in bound.layers {
    assert_eq(packing.layer, i)
    assert_eq(allocation, i + 6)
  }
}

///|
test "typed denoiser join rejects missing duplicate and misplaced components" {
  let packings = denoiser_test_packings()
  let entries = packings.mapi((i, p) => (p, i))
  let duplicate = entries[:].to_owned()
  duplicate.push(entries[0])
  let misplaced = entries[:].to_owned()
  misplaced.push((TextEncoder, 99))
  for wrong in [entries[:55].to_owned(), duplicate, misplaced] {
    try denoiser_bindings(wrong[:], 50, 2) catch {
      InvalidPacking => ()
      _ => fail("wrong join failure")
    } noraise {
      _ => fail("incomplete or ambiguous bindings accepted")
    }
  }
  try denoiser_bindings(entries[:], 49, 2) catch {
    InvalidPacking => ()
    _ => fail("wrong layer failure")
  } noraise {
    _ => fail("wrong model depth accepted")
  }
}

///|
test "denoiser borrow rejects released lease before any device access" {
  let owner = WeightStartup::new(max_device_bytes=1L)
  let lease : WeightLease = { owner, released: true, }
  let metadata = @h3.MiniMaxH3ModelMetadata::from_verified_content(
    @h3.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    @spec.ContentDigest::from_sha256(String::make(64, '8')),
  )
  try
    lease.with_denoiser_weights(Fl2vaDenoiser, @minimax_h3.build(metadata), (
      _,
      _,
      _,
    ) => ())
  catch {
    InvalidState => ()
    _ => fail("wrong lease failure")
  } noraise {
    _ => fail("released lease borrowed")
  }
  owner.close()
}
