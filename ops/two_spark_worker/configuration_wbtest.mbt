///|
test "worker deployment parsing rejects duplicate fields and unknown schema" {
  for
    source in [
      "{\"schema\":\"lunaflux.two-spark-worker.v1\",\"schema\":\"lunaflux.two-spark-worker.v1\"}",
      "{\"schema\":\"unknown\"}", "[]",
    ] {
    let bytes = @utf8.encode(source)
    try document(bytes, "1".repeat(64)) catch {
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
test "worker deployment label does not replace semantic parsing" {
  try document(b"{}", "0".repeat(64)) catch {
    TwoSparkWorkerError::Configuration => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("invalid deployment semantics accepted")
  }
}

///|
async test "referenced startup bytes are read once without checksum validation" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-spark-label")
    let path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(path, recursive=true))
    }
    @fs.write_file(
      path + "/source.bin",
      b"changed source",
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(path)
    defer root.close()
    let reference = Json::object({
      "path": Json::string("source.bin"),
      "sha256": Json::string("1".repeat(64)),
    })
    assert_eq(pinned(root, reference, 64L), b"changed source")
    try pinned(root, reference, 2L) catch {
      _ => ()
    } noraise {
      _ => fail("file-size limit ignored")
    }
  }
}
