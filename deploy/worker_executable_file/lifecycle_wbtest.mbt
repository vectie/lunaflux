///|
#borrow(handle)
extern "c" fn raw_test_close_while_active(
  handle : @cap.ApprovedExecutableHandle,
) -> Int = "lunaflux_worker_executable_test_close_while_active"

///|
async test "active spawn lease makes executable close retryably busy" {
  assert_true(@native_probe.linked())
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-executable-busy")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let bytes = @fs.read_file("/bin/cat").binary()
    let path = "\{root}/worker"
    @fs.write_file(path, bytes, create_mode=CreateNew)
    @fs.chmod(path, 0o500)
    let digest = WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    let admission = verify(
      path,
      digest,
      WorkerExecutableFileLimits::new(maximum_bytes=2097152L),
    )
    assert_eq(raw_test_close_while_active(admission.handle), 7)
    assert_true(admission.is_activation_live())
    admission.close()
    admission.close()
    assert_false(admission.is_activation_live())
  }
}
