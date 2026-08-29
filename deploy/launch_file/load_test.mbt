///|
async test "launch file is digest pinned and authority closed before publication" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-launch-file")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let text = "{\"schema\":\"lunaflux.launch.v1\",\"model_root\":\"/opt/lunaflux/model\",\"kernel_root\":\"/opt/lunaflux/kernels\",\"policy_root\":\"/opt/lunaflux/policy\",\"runtime_descriptor\":{\"locator\":\"runtime.json\",\"sha256\":\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"instance_policy\":{\"locator\":\"instance.json\",\"sha256\":\"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"worker_executable\":{\"path\":\"/opt/lunaflux/bin/worker\",\"sha256\":\"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"luna_approval\":{\"mode\":\"none\"}}"
    let bytes = @utf8.encode(text)
    let digest = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
    @fs.write_file("\{root}/lunaflux.launch.json", bytes, create_mode=CreateNew)
    let admitted = load_argument(
      "\{root}#sha256=\{digest}",
      try! LaunchFileLimits::new(maximum_bytes=65536L, maximum_json_depth=8),
    )
    assert_eq(admitted.digest().as_hex(), digest)
    @fs.rename(root, "\{root}.closed")
    @fs.rename("\{root}.closed", root)
    try
      load_argument(
        "\{root}#sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        try! LaunchFileLimits::new(maximum_bytes=65536L, maximum_json_depth=8),
      )
    catch {
      DigestMismatch => ()
      _ => fail("unexpected independent digest failure")
    } noraise {
      _ => fail("wrong independent launch digest was admitted")
    }
  }
}
