///|
pub fn open_root(path : Bytes, status : Ref[Int]) -> ApprovedHandle {
  raw_open_root(path, status)
}

///|
pub fn require_absolute_identity(root : ApprovedHandle, path : Bytes) -> Int {
  raw_require_absolute_identity(root, path)
}

///|
pub fn open_file(
  root : ApprovedHandle,
  locator : Bytes,
  status : Ref[Int],
) -> ApprovedHandle {
  raw_open_file(root, locator, status)
}

///|
pub fn close(handle : ApprovedHandle) -> Int {
  raw_close(handle)
}

///|
pub fn is_closed(handle : ApprovedHandle) -> Bool {
  raw_is_closed(handle)
}

///|
pub fn read_exact_at(
  file : ApprovedHandle,
  destination : FixedArray[Byte],
  destination_offset : Int,
  byte_count : Int,
  position : Int64,
) -> Int {
  raw_read_exact_at(file, destination, destination_offset, byte_count, position)
}

///|
pub fn stamp(file : ApprovedHandle, output : FixedArray[Int64]) -> Int {
  raw_stamp(file, output)
}

///|
pub fn read_immutable_snapshot(
  file : ApprovedHandle,
  maximum_bytes : Int64,
  status : Ref[Int],
) -> Bytes {
  raw_read_immutable_snapshot(file, maximum_bytes, b"", status)
}

///|
pub fn acquire_worker_roots(
  model_root : ApprovedHandle,
  kernel_root : ApprovedHandle,
  status : Ref[Int],
) -> @inheritance.WorkerApprovedRoots {
  raw_acquire_worker_roots(model_root, kernel_root, status)
}

///|
pub fn prepare_worker_roots() -> @inheritance.WorkerApprovedRoots {
  raw_prepare_worker_roots()
}

///|
pub fn acquire_prepared_worker_roots(
  roots : @inheritance.WorkerApprovedRoots,
  model_root : ApprovedHandle,
  kernel_root : ApprovedHandle,
) -> Int {
  raw_acquire_prepared_worker_roots(roots, model_root, kernel_root)
}

///|
pub fn duplicate_worker_root(
  roots : @inheritance.WorkerApprovedRoots,
  role : Int,
  status : Ref[Int],
) -> ApprovedHandle {
  raw_duplicate_worker_root(roots, role, status)
}

///|
pub fn worker_root_spawn_authority(
  roots : @inheritance.WorkerApprovedRoots,
) -> @inheritance.WorkerApprovedRootSpawnAuthority {
  raw_worker_root_spawn_authority(roots)
}

///|
pub fn open_inherited_root(role : Int, status : Ref[Int]) -> ApprovedHandle {
  raw_open_inherited_root(role, status)
}

///|
pub fn worker_roots_is_closed(roots : @inheritance.WorkerApprovedRoots) -> Bool {
  raw_worker_roots_is_closed(roots)
}

///|
pub fn close_worker_roots(roots : @inheritance.WorkerApprovedRoots) -> Int {
  raw_close_worker_roots(roots)
}
