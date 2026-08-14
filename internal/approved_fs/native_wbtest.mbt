///|
#borrow(path)
extern "c" fn raw_test_mkfifo(path : Bytes) -> Int = "lunaflux_approved_fs_test_mkfifo"

///|
#borrow(handle)
extern "c" fn raw_test_close_while_active(
  handle : ApprovedHandle,
  expected_kind : Int,
) -> Int = "lunaflux_approved_fs_test_close_while_active"

///|
#borrow(roots)
extern "c" fn raw_test_close_worker_roots_while_active(
  roots : @inheritance.WorkerApprovedRoots,
) -> Int = "lunaflux_approved_fs_test_close_worker_roots_while_active"

///|
extern "c" fn raw_test_reduce_identity_close(
  primary_status : Int,
  close_status : Int,
) -> Int = "lunaflux_approved_fs_test_reduce_identity_close"

///|
test "identity close reducer preserves the primary result" {
  assert_eq(raw_test_reduce_identity_close(10, 6), 10)
  assert_eq(raw_test_reduce_identity_close(1, 6), 1)
  assert_eq(raw_test_reduce_identity_close(0, 6), 6)
  assert_eq(raw_test_reduce_identity_close(0, 0), 0)
}

///|
async test "worker root leases remain reusable after Busy close" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-worker-roots")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let status = Ref(-1)
    let model = open_root(@utf8.encode(root_path), status)
    assert_eq(status.val, 0)
    let kernel = open_root(@utf8.encode(root_path), status)
    assert_eq(status.val, 0)
    let roots = acquire_worker_roots(model, kernel, status)
    assert_eq(status.val, 0)
    assert_eq(raw_test_close_worker_roots_while_active(roots), 7)
    assert_false(worker_roots_is_closed(roots))
    assert_eq(close_worker_roots(roots), 0)
    assert_true(worker_roots_is_closed(roots))
    assert_eq(close_worker_roots(roots), 0)
    assert_eq(close(model), 0)
    assert_eq(close(kernel), 0)
  }
}

///|
async test "FIFO final component is rejected without blocking" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-fifo")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    assert_eq(raw_test_mkfifo(@utf8.encode("\{root_path}/stream")), 0)
    let status = Ref(-1)
    let root = open_root(@utf8.encode(root_path), status)
    assert_eq(status.val, 0)
    let file = open_file(root, b"stream", status)
    assert_eq(status.val, 3)
    assert_true(is_closed(file))
    assert_eq(close(file), 0)
    assert_eq(close(root), 0)
  }
}

///|
async test "explicit close soak consumes every native owner" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-fds")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file("\{root_path}/file.bin", b"x", create_mode=CreateOrTruncate)
    for _ in 0..<256 {
      let status = Ref(-1)
      let root = open_root(@utf8.encode(root_path), status)
      assert_eq(status.val, 0)
      let file = open_file(root, b"file.bin", status)
      assert_eq(status.val, 0)
      assert_eq(close(root), 0)
      assert_eq(close(file), 0)
      assert_true(is_closed(root))
      assert_true(is_closed(file))
    }
  }
}

///|
async test "native ABI rejects invalid buffers without sentinel mutation" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-abi")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file(
      "\{root_path}/file.bin",
      b"abc",
      create_mode=CreateOrTruncate,
    )
    let status = Ref(-1)
    let root = open_root(@utf8.encode(root_path), status)
    assert_eq(status.val, 0)
    let escaped = open_file(root, b"../file.bin", status)
    assert_eq(status.val, 1)
    assert_true(is_closed(escaped))
    assert_eq(close(escaped), 0)
    let aliased = open_file(root, b"sub//file.bin", status)
    assert_eq(status.val, 1)
    assert_true(is_closed(aliased))
    assert_eq(close(aliased), 0)
    let file = open_file(root, b"file.bin", status)
    assert_eq(status.val, 0)
    let destination = FixedArray::make(3, b'!')
    assert_eq(read_exact_at(file, destination, 2, 2, 0L), 1)
    assert_eq(read_exact_at(file, destination, -1, 1, 0L), 1)
    assert_eq(read_exact_at(file, destination, 0, 1, 0x7fffffffffffffffL), 1)
    assert_eq(destination, FixedArray::make(3, b'!'))
    let stamp_output = FixedArray::make(4, 77L)
    assert_eq(stamp(file, stamp_output), 1)
    assert_eq(stamp_output, FixedArray::make(4, 77L))
    let snapshot_status = Ref(-1)
    assert_eq(read_immutable_snapshot(file, -1L, snapshot_status), b"")
    assert_eq(snapshot_status.val, 1)
    assert_eq(read_immutable_snapshot(file, 2L, snapshot_status), b"")
    assert_eq(snapshot_status.val, 9)
    assert_eq(read_immutable_snapshot(file, 3L, snapshot_status), b"abc")
    assert_eq(snapshot_status.val, 0)
    assert_eq(close(root), 0)
    assert_eq(close(file), 0)
  }
}

///|
async test "active operations make close retryable without consuming authority" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-busy")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file("\{root_path}/file.bin", b"x", create_mode=CreateOrTruncate)
    let status = Ref(-1)
    let root = open_root(@utf8.encode(root_path), status)
    assert_eq(status.val, 0)
    let file = open_file(root, b"file.bin", status)
    assert_eq(status.val, 0)
    assert_eq(raw_test_close_while_active(root, 1), 7)
    assert_false(is_closed(root))
    assert_eq(raw_test_close_while_active(file, 2), 7)
    assert_false(is_closed(file))
    assert_eq(close(root), 0)
    assert_eq(close(file), 0)
  }
}
