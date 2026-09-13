///|
test "typed integer parsing preserves bounds and canonical spelling" {
  let reader : LunaTensorCoreEvidenceReader = {
    lines: [
      "n=2147483647", "n=-2147483648", "n=9223372036854775807", "n=-9223372036854775808",
      "",
    ],
    cursor: 0,
  }
  assert_eq(reader.read_int("n"), 2147483647)
  assert_eq(reader.read_int("n"), -2147483648)
  assert_eq(reader.read_int64("n"), 9223372036854775807L)
  assert_eq(reader.read_int64("n"), -9223372036854775808L)
  for value in ["2147483648", "-2147483649", "01", "+1", "1.0", "1e0", ""] {
    let reader : LunaTensorCoreEvidenceReader = {
      lines: ["n=" + value, ""],
      cursor: 0,
    }
    try reader.read_int("n") catch {
      LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument) => ()
      _ => fail("unexpected parse error")
    } noraise {
      _ => fail("accepted invalid integer: " + value)
    }
  }
  for
    value in [
      "9223372036854775808", "-9223372036854775809", "01", "+1", "1.0", "",
    ] {
    let reader : LunaTensorCoreEvidenceReader = {
      lines: ["n=" + value, ""],
      cursor: 0,
    }
    try reader.read_int64("n") catch {
      LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument) => ()
      _ => fail("unexpected parse error")
    } noraise {
      _ => fail("accepted invalid integer: " + value)
    }
  }
}
