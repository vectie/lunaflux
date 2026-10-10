///|
async test "offline observations consume identity labels without payload authentication" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-offline-label")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let path = directory + "/observations"
    let bytes = b"supplied offline observations"
    @fs.write_file(path, bytes, create_mode=CreateNew)
    assert_eq(
      load_attention_offline_input(
        Some((path, "f".repeat(64), "device")),
        "toolchain",
      ),
      Some((bytes, "device", "toolchain")),
    )
    assert_eq(load_attention_offline_input(None, "toolchain"), None)
    try
      load_attention_offline_input(
        Some((path + ".missing", "f".repeat(64), "device")),
        "toolchain",
      )
    catch {
      InvalidArguments => ()
      e => raise e
    } noraise {
      _ => fail("unavailable observations accepted")
    }
  }
}
