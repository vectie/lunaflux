///|
test "blockwise storage distinguishes retained validity from shared cells" {
  let retained = @attention_physical_ir.BlockwiseTileStorage::new(
    dimensions=128,
    keys=32,
    query_consumers=2,
    stages=2,
    validity_slots=0,
  ).unwrap()
  assert_eq(retained.region_offsets_bytes(), (0L, 32768L, 33024L))
  assert_eq(retained.region_sizes_bytes(), (32768L, 256L, 0L))
  assert_eq(retained.minimum_bytes(), 33024L)
  assert_false(retained.accepts_reservation(33023L))
  assert_true(retained.accepts_reservation(33024L))
  assert_true(retained.accepts_reservation(33040L))
  assert_false(retained.accepts_reservation(2147483648L))
  let shared = @attention_physical_ir.BlockwiseTileStorage::new(
    dimensions=128,
    keys=32,
    query_consumers=2,
    stages=2,
    validity_slots=2,
  ).unwrap()
  assert_eq(shared.region_offsets_bytes(), retained.region_offsets_bytes())
  assert_eq(shared.minimum_bytes(), 33032L)
  assert_false(shared.accepts_reservation(33024L))
  assert_true(shared.accepts_reservation(33032L))
  // Reservation is not live storage: a backend may still request 33040 bytes.
  assert_true(shared.minimum_bytes() <= 33040L)
}

///|
test "blockwise storage covers non-pipelined and larger ownership domains" {
  let one = @attention_physical_ir.BlockwiseTileStorage::new(
    dimensions=64,
    keys=64,
    query_consumers=4,
    stages=1,
    validity_slots=1,
  ).unwrap()
  assert_eq(one.region_offsets_bytes(), (0L, 16384L, 17408L))
  assert_eq(one.minimum_bytes(), 17412L)
}

///|
test "blockwise storage rejects overflow and unsupported region ownership" {
  for dimensions in [0, -1, 127, 2147483646] {
    assert_eq(
      @attention_physical_ir.BlockwiseTileStorage::new(
        dimensions~,
        keys=32,
        query_consumers=2,
        stages=2,
        validity_slots=0,
      ),
      None,
    )
  }
  for
    (keys, query_consumers, stages, validity_slots) in [
      (0, 2, 2, 0),
      (32, 0, 2, 0),
      (32, 2, 0, 0),
      (32, 2, 3, 0),
      (32, 2, 2, -1),
      (32, 2, 2, 3),
      (2147483647, 2, 2, 0),
      (32, 2147483647, 2, 0),
    ] {
    assert_eq(
      @attention_physical_ir.BlockwiseTileStorage::new(
        dimensions=128,
        keys~,
        query_consumers~,
        stages~,
        validity_slots~,
      ),
      None,
    )
  }
}
