///|
async test "worker executable snapshot binds exact path digest and closes owners" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-executable")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let path = "\{root}/worker"
    let bytes = b"worker fixture"
    @fs.write_file(path, bytes, create_mode=CreateNew)
    @fs.chmod(path, 0o500)
    let digest = WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    let admission = verify(
      path,
      digest,
      try! WorkerExecutableFileLimits::new(maximum_bytes=4096L),
    )
    assert_eq(admission.digest(), digest)
    assert_eq(@utf8.decode(admission.activation_path()), path)
    @fs.rename(path, "\{root}/worker.moved")
    assert_true(admission.is_activation_live())
    admission.close()
    assert_false(admission.is_activation_live())
  }
}

///|
async test "worker executable rejects directories bad paths and digest drift" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-executable-hostile")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    @fs.mkdir("\{root}/directory")
    let digest = try! WorkerExecutableDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    for path in ["relative", "\{root}/../worker", "\{root}/directory"] {
      try
        verify(
          path,
          digest,
          try! WorkerExecutableFileLimits::new(maximum_bytes=4096L),
        )
      catch {
        InvalidPath | UnsupportedFileType | FileUnavailable => ()
        _ => fail("unexpected hostile executable failure")
      } noraise {
        _ => fail("hostile executable path was admitted")
      }
    }
  }
}

///|
async test "materialized worker verifies staged bytes for exact target path" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-materialized")
    let source_label = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(source_label, recursive=true))
    }
    let bytes = b"materialized worker fixture"
    @fs.write_file(
      "\{source_label}/lunaflux-device-worker",
      bytes,
      create_mode=CreateNew,
    )
    let digest = WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    let source_root = @approved_fs.ApprovedRoot::open_absolute(source_label)
    let source = source_root.materialization_view(
      source_label, "/opt/lunaflux/bin",
    )
    let admission = verify_materialized(
      source,
      "lunaflux-device-worker",
      "/opt/lunaflux/bin/lunaflux-device-worker",
      digest,
      try! WorkerExecutableFileLimits::new(maximum_bytes=4096L),
    )
    assert_eq(admission.digest(), digest)
    assert_eq(
      @utf8.decode(admission.activation_path()),
      "/opt/lunaflux/bin/lunaflux-device-worker",
    )
    try
      verify_materialized(
        source,
        "lunaflux-device-worker",
        "/usr/bin/lunaflux-device-worker",
        digest,
        try! WorkerExecutableFileLimits::new(maximum_bytes=4096L),
      )
    catch {
      InvalidPath => ()
      _ => fail("target substitution failed at the wrong boundary")
    } noraise {
      _ => fail("target substitution was admitted")
    }
    source_root.close()
  }
}

///|
#borrow(source, destination)
extern "c" fn raw_test_link(source : Bytes, destination : Bytes) -> Int = "lunaflux_worker_executable_test_link"

///|
async test "pinned executable survives replacement hardlink and descriptor reuse" {
  assert_true(@native_probe.linked())
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-executable-toctou")
    let root_label = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_label, recursive=true))
    }
    let cat_bytes = @fs.read_file("/bin/cat").binary()
    let false_bytes = @fs.read_file("/bin/sh").binary()
    let path = "\{root_label}/worker"
    let retained_link = "\{root_label}/authenticated-hardlink"
    @fs.write_file(path, cat_bytes, create_mode=CreateNew)
    @fs.chmod(path, 0o500)
    let digest = WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(cat_bytes)),
    )
    let limits = WorkerExecutableFileLimits::new(maximum_bytes=2097152L)
    let admission = verify(path, digest, limits)
    assert_eq(raw_test_link(@utf8.encode(path), @utf8.encode(retained_link)), 0)
    @fs.rename(path, "\{root_label}/renamed-original")
    @fs.write_file(path, false_bytes, create_mode=CreateNew)
    @fs.chmod(path, 0o500)

    let process_limits = @process.ProcessLimits::new(
      max_frame_bytes=64,
      io_timeout_millis=1000,
      shutdown_timeout_millis=1000,
    )
    let prepared = try admission.prepare_child(process_limits) catch {
      UnsupportedPlatform => None
      _ => fail("pinned executable preparation failed unexpectedly")
    } noraise {
      child => Some(child)
    }
    match prepared {
      None => ()
      Some(prepared) => {
        let root = @approved_fs.ApprovedRoot::open_absolute(root_label)
        let roots = @approved_fs.acquire_worker_approved_roots(
          model_root=root,
          kernel_root=root,
        )
        let owner = prepared.spawn_with_approved_roots(
          @approved_fs.worker_approved_root_spawn_authority(roots),
        )
        let source = FixedArray::make(4, b'x')
        let destination = FixedArray::make(4, b'!')
        owner.write_frame(source, offset=0, byte_count=4)
        assert_eq(owner.read_frame(destination), 4)
        assert_eq(destination, source)
        owner.close()
        @approved_fs.close_worker_approved_roots(roots)
        root.close()
      }
    }
    admission.close()
    assert_false(admission.is_activation_live())
    for _ in 0..<16 {
      let replacement = verify(
        path,
        WorkerExecutableDigest::from_sha256(
          @crypto.bytes_to_hex_string(@crypto.sha256(false_bytes)),
        ),
        limits,
      )
      replacement.close()
    }
    try admission.prepare_child(process_limits) catch {
      FileUnavailable => ()
      _ => fail("closed executable failed with the wrong lifecycle error")
    } noraise {
      _ => fail("closed executable survived descriptor reuse")
    }
  }
}

///|
async test "executable admission rejects final and intermediate symlinks" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-worker-executable-symlink")
    let root_label = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_label, recursive=true))
    }
    let bytes = @fs.read_file("/bin/cat").binary()
    let real_path = "\{root_label}/worker"
    @fs.write_file(real_path, bytes, create_mode=CreateNew)
    @fs.chmod(real_path, 0o500)
    @fs.symlink(target=real_path, "\{root_label}/worker-link")
    @fs.mkdir("\{root_label}/real-parent")
    @fs.symlink(target="\{root_label}/real-parent", "\{root_label}/parent-link")
    let digest = WorkerExecutableDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    )
    let limits = WorkerExecutableFileLimits::new(maximum_bytes=2097152L)
    for
      path in ["\{root_label}/worker-link", "\{root_label}/parent-link/worker"] {
      try verify(path, digest, limits) catch {
        UnsupportedFileType | FileUnavailable => ()
        _ => fail("symlink failed at the wrong executable boundary")
      } noraise {
        _ => fail("symlink executable was admitted")
      }
    }
  }
}
