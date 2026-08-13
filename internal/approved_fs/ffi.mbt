///|
#borrow(path, status)
extern "c" fn raw_open_root(
  path : Bytes,
  status : Ref[Int],
) -> @cap.ApprovedHandle = "lunaflux_approved_fs_open_root"

///|
#borrow(root, locator, status)
extern "c" fn raw_open_file(
  root : @cap.ApprovedHandle,
  locator : Bytes,
  status : Ref[Int],
) -> @cap.ApprovedHandle = "lunaflux_approved_fs_open_file"

///|
#borrow(handle)
extern "c" fn raw_close(handle : @cap.ApprovedHandle) -> Int = "lunaflux_approved_fs_close"

///|
#borrow(handle)
extern "c" fn raw_is_closed(handle : @cap.ApprovedHandle) -> Bool = "lunaflux_approved_fs_is_closed"

///|
#borrow(file, destination)
extern "c" fn raw_read_exact_at(
  file : @cap.ApprovedHandle,
  destination : FixedArray[Byte],
  destination_offset : Int,
  byte_count : Int,
  position : Int64,
) -> Int = "lunaflux_approved_fs_read_exact_at"

///|
#borrow(file, output)
extern "c" fn raw_stamp(
  file : @cap.ApprovedHandle,
  output : FixedArray[Int64],
) -> Int = "lunaflux_approved_fs_stamp"

///|
#borrow(file, failure, status)
extern "c" fn raw_read_immutable_snapshot(
  file : @cap.ApprovedHandle,
  maximum_bytes : Int64,
  failure : Bytes,
  status : Ref[Int],
) -> Bytes = "lunaflux_approved_fs_read_immutable_snapshot"
