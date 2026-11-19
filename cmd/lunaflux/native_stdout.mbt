///|
extern "c" fn raw_native_stdout_flush() -> Int32 = "lunaflux_native_stdout_flush"

///|
fn native_stdout_flush_succeeded(status : Int32) -> Bool {
  status == 0
}

///|
fn flush_native_stdout_after_readiness_publication() -> Bool {
  native_stdout_flush_succeeded(raw_native_stdout_flush())
}
