///|
async test "bridge reads a supplied tokenizer label without checksum scanning" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-bridge-label")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let path = "\{directory}/tokenizer.json"
    @fs.write_file(path, b"opaque tokenizer", create_mode=CreateNew)
    let argument = path + "#sha256=" + "f".repeat(64)
    assert_eq(bridge_read_file(argument, 32, "tokenizer"), b"opaque tokenizer")
    try bridge_read_file(argument, 4, "tokenizer") catch {
      StartupIdentity(_) => ()
      _ => fail("unexpected bridge size failure")
    } noraise {
      _ => fail("oversized tokenizer was readmitted")
    }
    let empty = "\{directory}/empty.json"
    @fs.write_file(empty, b"", create_mode=CreateNew)
    try
      bridge_read_file(empty + "#sha256=" + "f".repeat(64), 32, "tokenizer")
    catch {
      StartupIdentity(_) => ()
      _ => fail("unexpected empty tokenizer failure")
    } noraise {
      _ => fail("empty tokenizer was admitted")
    }
  }
}
