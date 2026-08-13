///|
priv type NativeChild

///|
#borrow(path, status)
extern "c" fn raw_spawn(path : Bytes, status : Ref[Int]) -> NativeChild = "lunaflux_process_spawn"

///|
#borrow(path, roots, status)
extern "c" fn raw_spawn_with_approved_roots(
  path : Bytes,
  roots : @inheritance.WorkerApprovedRoots,
  status : Ref[Int],
) -> NativeChild = "lunaflux_process_spawn_with_approved_roots"

///|
#borrow(child, source)
extern "c" fn raw_write_exact(
  child : NativeChild,
  source : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
  timeout_millis : Int,
) -> Int = "lunaflux_process_write_exact"

///|
#borrow(child, destination)
extern "c" fn raw_read_exact(
  child : NativeChild,
  destination : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
  timeout_millis : Int,
) -> Int = "lunaflux_process_read_exact"

///|
#borrow(child)
extern "c" fn raw_shutdown_write(child : NativeChild) -> Int = "lunaflux_process_shutdown_write"

///|
#borrow(child, kind, code)
extern "c" fn raw_wait(
  child : NativeChild,
  timeout_millis : Int,
  kind : Ref[Int],
  code : Ref[Int],
) -> Int = "lunaflux_process_wait"

///|
#borrow(child)
extern "c" fn raw_terminate(child : NativeChild) -> Int = "lunaflux_process_terminate"

///|
#borrow(child)
extern "c" fn raw_close(child : NativeChild) -> Int = "lunaflux_process_close"

///|
#borrow(source)
extern "c" fn raw_inherited_write_exact(
  source : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
  timeout_millis : Int,
) -> Int = "lunaflux_process_inherited_write_exact"

///|
#borrow(destination)
extern "c" fn raw_inherited_read_exact(
  destination : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
  timeout_millis : Int,
) -> Int = "lunaflux_process_inherited_read_exact"
