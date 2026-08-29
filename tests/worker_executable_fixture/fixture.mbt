///|
pub(all) suberror WorkerExecutableFixtureError

///|
#borrow(path, status)
extern "c" fn read_fixture(path : Bytes, status : Ref[Int]) -> Bytes = "lunaflux_worker_executable_fixture_read"

///|
extern "c" fn raw_cat_path() -> Bytes = "lunaflux_worker_executable_fixture_cat_path"

///|
/// Exact test executable path whose parent components are real directories on
/// the current host. Production admission must continue rejecting symlinks.
pub fn cat_path() -> Bytes {
  raw_cat_path()
}

///|
/// Test helper that deliberately traverses the production verifier. On a host
/// without descriptor-pinned execution, later child preparation fails closed.
pub fn admit(
  executable : Bytes,
) -> @worker_executable_file.WorkerExecutableAdmission raise WorkerExecutableFixtureError {
  let digest_hex = digest(executable)
  let admitted_digest = @worker_executable_file.WorkerExecutableDigest::from_sha256(
    digest_hex,
  ) catch {
    _ => raise WorkerExecutableFixtureError
  }
  let path = @utf8.decode(executable) catch {
    _ => raise WorkerExecutableFixtureError
  }
  @worker_executable_file.verify(
    path,
    admitted_digest,
    @worker_executable_file.WorkerExecutableFileLimits::new(
      maximum_bytes=67108864L,
    ),
  ) catch {
    _ => raise WorkerExecutableFixtureError
  }
}

///|
pub fn digest(executable : Bytes) -> String raise WorkerExecutableFixtureError {
  let status = Ref(1)
  let bytes = read_fixture(executable, status)
  if status.val != 0 {
    raise WorkerExecutableFixtureError
  }
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
/// Test-only spawn convenience backed by the production pinned admission.
/// Repository boundary gates forbid production packages from importing it.
pub fn spawn_with_approved_roots(
  executable : Bytes,
  limits : @process.ProcessLimits,
  roots : @approved_fs.WorkerApprovedRoots,
) -> @process.ChildProcess raise WorkerExecutableFixtureError {
  let admission = admit(executable)
  let child = try
    admission
    .prepare_child(limits)
    .spawn_with_approved_roots(
      @approved_fs.worker_approved_root_spawn_authority(roots),
    )
  catch {
    _ => {
      try admission.close() catch {
        _ => ()
      } noraise {
        _ => ()
      }
      raise WorkerExecutableFixtureError
    }
  } noraise {
    value => value
  }
  try admission.close() catch {
    _ => {
      try child.close() catch {
        _ => ()
      } noraise {
        _ => ()
      }
      raise WorkerExecutableFixtureError
    }
  } noraise {
    _ => ()
  }
  child
}
