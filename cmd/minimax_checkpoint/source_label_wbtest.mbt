///|
test "checkpoint location consumes a declared identity without config hashing" {
  let label = "d".repeat(64)
  let (root, identity) = checkpoint_location("/model/FL2VA#label=" + label)
  assert_eq(root, "/model/FL2VA")
  assert_eq(identity.unwrap().as_hex(), label)
  assert_eq(checkpoint_location("/model/FL2VA"), ("/model/FL2VA", None))
  for
    input in [
      "/model#label=invalid",
      "#label=" + label,
      "/model#label=" + label + "#label=" + label,
    ] {
    try checkpoint_location(input) catch {
      _ => ()
    } noraise {
      _ => fail("invalid declared location accepted")
    }
  }
}

///|
test "inventory labels are associations not payload checksum claims" {
  let label = "e".repeat(64)
  for prefix in ["label:" + label + "  ", label + "  weight"] {
    assert_eq(inventory_prefix_label(@utf8.encode(prefix)[:]).as_hex(), label)
  }
  for prefix in ["", "label:" + "g".repeat(64) + "  ", "label:" + label + "!!"] {
    try inventory_prefix_label(@utf8.encode(prefix)[:]) catch {
      _ => ()
    } noraise {
      _ => fail("invalid inventory prefix accepted")
    }
  }
}

///|
async test "checkpoint identity reads only the supplied inventory prefix" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-h3-label")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let label = "b".repeat(64)
    let prefix = @utf8.encode("label:" + label + "  ")
    let bytes = FixedArray::make(8192, b'\xff')
    for i, byte in prefix {
      bytes[i] = byte
    }
    @fs.write_file(
      canonical + "/TEXT.inventory",
      Bytes::from_array(bytes[:]),
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    assert_eq(inventory_label(root, "TEXT.inventory").unwrap().as_hex(), label)
    assert_eq(inventory_label(root, "absent.inventory"), None)
    // No model config or tensor payload exists in this fixture. Identity
    // association must not read or authenticate either of them.
  }
}
