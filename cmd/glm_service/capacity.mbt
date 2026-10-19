///|
/// Preparation ownership is independent of resident GPU slots. The old recipe
/// keeps two lanes per slot; deployments may choose fewer waiting owners.
fn service_admission_lanes(json : Json, slots : Int) -> Int raise {
  guard json is Object(fields) else { fail("service object required") }
  let lanes = if fields.contains("admission_lanes") {
    json_int(json,"admission_lanes")
  } else { slots * 2 }
  guard slots > 0 && lanes >= slots && lanes <= 1024 else {
    fail("invalid admission lane capacity")
  }
  lanes
}

///|
test "preparation and connection ownership need not duplicate resident slots" {
  assert_eq(service_admission_lanes({},4),8)
  assert_eq(service_admission_lanes({"admission_lanes":16},16),16)
  try service_admission_lanes({"admission_lanes":15},16) catch { _ => () }
  noraise { _ => fail("preparation cannot expose the declared resident concurrency") }
}

///|
/// A deployment chooses aggregate physical pages independently of its logical
/// context/slot envelope. Omitted pages preserve the existing service recipe.
fn service_history_capacity(
  json : Json,
  slots : Int,
  context : Int,
) -> @history.ResidentHistoryCapacity raise {
  guard slots > 0 &&
    slots <= 256 &&
    context >= 16 &&
    context <= 1048576 &&
    context % 16 == 0 else {
    fail("invalid service history geometry")
  }
  guard json is Object(fields) else { fail("service object required") }
  let pages = if fields.contains("history_pages") {
    json_int(json, "history_pages")
  } else {
    slots * (context / 16)
  }
  guard @history.ResidentHistoryCapacity::new(
      rows_per_page=16,
      physical_pages=pages,
      maximum_context=context,
      maximum_sequences=slots,
      decode_reserve_pages=1,
    )
    is Some(capacity) else {
    fail("shared history pool cannot fit maximum context and decode reserve")
  }
  capacity
}

///|
test "service shares an 8K history pool across sixteen slots without 16x KV" {
  let capacity = service_history_capacity({ "history_pages": 513 }, 16, 8192)
  assert_eq(capacity.domain().geometry(), (16, 513, 8192, 16))
  assert_eq(capacity.resident_sequences_at(512), Some(16))
  assert_eq(capacity.resident_sequences_at(8192), Some(1))
  assert_eq(
    service_history_capacity({}, 4, 2048).domain().physical_rows(),
    8192,
  )
  try service_history_capacity({ "history_pages": 512 }, 16, 8192) catch {
    _ => ()
  } noraise {
    _ => fail("accepted impossible full-length request")
  }
}
