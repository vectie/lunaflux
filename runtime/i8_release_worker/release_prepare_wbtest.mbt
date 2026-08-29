///|
priv struct ReleaseBlueprintMarker {
  identity : Int
}

///|
async test "release forwards exact blueprint object and admitted activation only" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-i8-release-executable")
    let directory = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let first_path = "\{directory}/worker-first"
    let second_path = "\{directory}/worker-second"
    let first_bytes = b"first admitted worker"
    let second_bytes = b"second admitted worker"
    @fs.write_file(first_path, first_bytes, create_mode=CreateNew)
    @fs.write_file(second_path, second_bytes, create_mode=CreateNew)
    @fs.chmod(first_path, 0o500)
    @fs.chmod(second_path, 0o500)
    let limits = try! @worker_executable_file.WorkerExecutableFileLimits::new(
      maximum_bytes=4096L,
    )
    let first_digest = @worker_executable_file.WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(first_bytes)),
    ) catch {
      _ => fail("first executable digest construction failed")
    }
    let first = @worker_executable_file.verify(first_path, first_digest, limits)
    let second_digest = @worker_executable_file.WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(second_bytes)),
    ) catch {
      _ => fail("second executable digest construction failed")
    }
    let second = @worker_executable_file.verify(
      second_path, second_digest, limits,
    )
    let blueprint = { identity: 7 }
    let retained = retain_exact_release_inputs(blueprint, first)
    assert_true(physical_equal(retained.blueprint, blueprint))
    assert_eq(retained.blueprint.identity, 7)
    assert_eq(retained.executable.activation_path(), first.activation_path())
    assert_false(
      retained.executable.activation_path() == second.activation_path(),
    )
  }
}
