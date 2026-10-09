///|
test "worker deployment parsing rejects duplicate fields and unknown schema" {
  for
    source in [
      "{\"schema\":\"lunaflux.two-spark-worker.v1\",\"schema\":\"lunaflux.two-spark-worker.v1\"}",
      "{\"schema\":\"unknown\"}", "[]",
    ] {
    let bytes = @utf8.encode(source)
    try
      document(bytes, @crypto.bytes_to_hex_string(@crypto.sha256(bytes)))
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid deployment accepted")
    }
  }
}

///|
test "worker memory values preserve exact integer precision and reject negatives" {
  let object : Map[String, Json] = { "bytes": Json::string("9007199254740993") }
  assert_eq(number(object, "bytes"), 9007199254740993L)
  object["bytes"] = Json::string("-1")
  try number(object, "bytes") catch {
    _ => ()
  } noraise {
    _ => fail("negative memory")
  }
}

///|
test "worker deployment digest is checked before interpreting fields" {
  try document(b"{}", "0".repeat(64)) catch {
    TwoSparkWorkerError::Digest => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("unpinned deployment")
  }
}
