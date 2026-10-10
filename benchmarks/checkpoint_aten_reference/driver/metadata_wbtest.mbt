///|
test "reference numeric extents reject fractional or inexact file offsets" {
  assert_eq(integer(Json::number(66714780128.0)), 66714780128L)
  for
    json in [
      Json::number(-1.0),
      Json::number(1.5),
      Json::number(9007199254740992.0),
      Json::string("1"),
    ] {
    try integer(json) catch {
      _ => ()
    } noraise {
      _ => fail("invalid original tensor extent must not be rounded")
    }
  }
}

///|
test "reference metadata requires original named fields and positive dimensions" {
  let document = @json.parse("{\"width\":5120,\"eps\":0.000001,\"invalid\":0}")
  assert_eq(extent(document, "width"), 5120)
  assert_eq(scalar(document, "eps"), 0.000001)
  try extent(document, "invalid") catch {
    _ => ()
  } noraise {
    _ => fail("zero width must fail")
  }
  try field(document, "missing") catch {
    _ => ()
  } noraise {
    _ => fail("missing checkpoint field must not use a made-up default")
  }
}
