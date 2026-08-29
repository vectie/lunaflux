///|
fn process_limits() -> @process.ProcessLimits raise {
  @process.ProcessLimits::new(
    max_frame_bytes=64,
    io_timeout_millis=5000,
    shutdown_timeout_millis=5000,
  )
}

///|
extern "c" fn raw_open_ambient_fd() -> Bool = "lunaflux_inheritance_test_open_ambient_fd"

///|
extern "c" fn raw_close_ambient_fd() -> Bool = "lunaflux_inheritance_test_close_ambient_fd"

///|
async fn run(executable : Bytes) -> Unit {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-root-inheritance")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let model_path = "\{parent}/model"
    let kernel_path = "\{parent}/kernel"
    @fs.mkdir(model_path)
    @fs.mkdir(kernel_path)
    @fs.write_file(
      "\{model_path}/sentinel.bin",
      b"m",
      create_mode=CreateOrTruncate,
    )
    @fs.write_file(
      "\{kernel_path}/sentinel.bin",
      b"k",
      create_mode=CreateOrTruncate,
    )
    assert_true(raw_open_ambient_fd())
    let model = @approved_fs.ApprovedRoot::open_absolute(model_path)
    let kernel = @approved_fs.ApprovedRoot::open_absolute(kernel_path)
    let roots = @approved_fs.acquire_worker_approved_roots(
      model_root=model,
      kernel_root=kernel,
    )
    model.close()
    kernel.close()
    let receive = FixedArray::make(8, b'\x00')
    for iteration in 0..<2 {
      let child = @worker_executable_fixture.spawn_with_approved_roots(
        executable,
        process_limits(),
        roots,
      )
      assert_eq(child.read_frame(receive), 8)
      assert_eq(
        receive,
        FixedArray::from_array([b'm', b'k', b'1', b'1', b'1', b'1', b'1', b'1']),
      )
      let exit = child.wait()
      assert_eq(exit.kind(), 0)
      assert_eq(exit.code(), 0)
      child.close()
      if iteration == 0 {
        @fs.rename(model_path, "\{parent}/old-model")
        @fs.rename(kernel_path, "\{parent}/old-kernel")
        @fs.mkdir(model_path)
        @fs.mkdir(kernel_path)
        @fs.write_file(
          "\{model_path}/sentinel.bin",
          b"x",
          create_mode=CreateOrTruncate,
        )
        @fs.write_file(
          "\{kernel_path}/sentinel.bin",
          b"y",
          create_mode=CreateOrTruncate,
        )
      }
    }
    assert_true(raw_close_ambient_fd())
    assert_false(@approved_fs.worker_approved_roots_is_closed(roots))
    @approved_fs.close_worker_approved_roots(roots)
    @approved_fs.close_worker_approved_roots(roots)
    assert_true(@approved_fs.worker_approved_roots_is_closed(roots))
    try
      @worker_executable_fixture.spawn_with_approved_roots(
        executable,
        process_limits(),
        roots,
      )
    catch {
      _ => ()
    } noraise {
      child => {
        child.close()
        fail("closed worker roots were inherited")
      }
    }
  }
}

///|
async fn main {
  match @env.args() {
    [_, executable] => run(@utf8.encode(executable))
    _ => println("approved-root child executable path is required")
  }
}
