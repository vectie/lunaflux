///|
extern "c" fn raw_fd_closed(fd : Int) -> Bool = "lunaflux_approved_root_probe_fd_closed"

///|
extern "c" fn raw_environment_empty() -> Bool = "lunaflux_approved_root_probe_environment_empty"

///|
extern "c" fn raw_directory_fds_cloexec() -> Bool = "lunaflux_approved_root_probe_directory_fds_cloexec"

///|
fn read_sentinel(root : @approved_fs.ApprovedRoot) -> Byte raise {
  let file = root.open_file(
    @approved_fs.ApprovedRelativeLocator::new("sentinel.bin"),
  )
  let bytes = FixedArray::make(1, b'\x00')
  file.read_exactly_at(bytes, destination_offset=0, byte_count=1, position=0L)
  file.close()
  bytes[0]
}

///|
fn main {
  try! ({
    let model_root = @approved_fs.ApprovedRoot::open_inherited(Model)
    let kernel_root = @approved_fs.ApprovedRoot::open_inherited(Kernel)
    let model = read_sentinel(model_root)
    let kernel = read_sentinel(kernel_root)
    let owned_cloexec = raw_directory_fds_cloexec()
    model_root.close()
    kernel_root.close()
    let channel = @process.open_inherited(
      @process.ProcessLimits::new(
        max_frame_bytes=64,
        io_timeout_millis=5000,
        shutdown_timeout_millis=5000,
      ),
    )
    let argv_fixed = match @env.args() {
      ["lunaflux-worker"] => true
      _ => false
    }
    let output = FixedArray::from_array([
      model,
      kernel,
      if argv_fixed {
        b'1'
      } else {
        b'0'
      },
      if raw_environment_empty() {
        b'1'
      } else {
        b'0'
      },
      if raw_fd_closed(2) {
        b'1'
      } else {
        b'0'
      },
      if raw_fd_closed(3) && raw_fd_closed(4) {
        b'1'
      } else {
        b'0'
      },
      if raw_fd_closed(9) {
        b'1'
      } else {
        b'0'
      },
      if owned_cloexec {
        b'1'
      } else {
        b'0'
      },
    ])
    channel.write_frame(output, offset=0, byte_count=8)
  })
}
