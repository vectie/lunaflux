///|
priv type NativeChild

///|
extern "c" fn raw_prepare_child() -> NativeChild = "lunaflux_process_prepare_child"

///|
#borrow(child, path)
extern "c" fn raw_spawn_prepared(child : NativeChild, path : Bytes) -> Int = "lunaflux_process_spawn_prepared"

///|
#borrow(child, path, roots)
extern "c" fn raw_spawn_prepared_with_approved_roots(
  child : NativeChild,
  path : Bytes,
  roots : @inheritance.WorkerApprovedRoots,
) -> Int = "lunaflux_process_spawn_prepared_with_approved_roots"

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
extern "c" fn raw_monotonic_now() -> Int64 = "lunaflux_process_monotonic_now"

///|
#borrow(child, source)
extern "c" fn raw_try_write(
  child : NativeChild,
  source : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
) -> Int = "lunaflux_process_try_write"

///|
#borrow(child, destination)
extern "c" fn raw_try_read(
  child : NativeChild,
  destination : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
) -> Int = "lunaflux_process_try_read"

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
#borrow(child)
extern "c" fn raw_is_closed(child : NativeChild) -> Bool = "lunaflux_process_is_closed"

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

///|
#borrow(prefix)
extern "c" fn raw_inherited_read_prefix_or_eof(
  prefix : FixedArray[Byte],
  timeout_millis : Int,
) -> Int = "lunaflux_process_inherited_read_prefix_or_eof"

///|
extern "c" fn raw_inherited_expect_clean_eof(timeout_millis : Int) -> Int = "lunaflux_process_inherited_expect_clean_eof"

///|
extern "c" fn raw_exit_failure() -> Unit = "lunaflux_process_exit_failure"
