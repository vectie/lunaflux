///|
#borrow(path, status)
extern "c" fn raw_open_executable(
  path : Bytes,
  maximum_bytes : Int64,
  status : Ref[Int],
) -> @cap.ApprovedExecutableHandle = "lunaflux_worker_executable_open"

///|
#borrow(handle, failure, status)
extern "c" fn raw_snapshot_and_pin(
  handle : @cap.ApprovedExecutableHandle,
  maximum_bytes : Int64,
  failure : Bytes,
  status : Ref[Int],
) -> Bytes = "lunaflux_worker_executable_snapshot_and_pin"

///|
#borrow(handle)
extern "c" fn raw_close_executable(
  handle : @cap.ApprovedExecutableHandle,
) -> Int = "lunaflux_worker_executable_close"

///|
#borrow(handle)
extern "c" fn raw_executable_is_closed(
  handle : @cap.ApprovedExecutableHandle,
) -> Bool = "lunaflux_worker_executable_is_closed"

///|
#borrow(handle)
extern "c" fn raw_executable_spawn_supported(
  handle : @cap.ApprovedExecutableHandle,
) -> Bool = "lunaflux_worker_executable_spawn_supported"
