///|
extern "c" fn raw_inherited_wait(
  write_mode : Bool,
  timeout_millis : Int32,
) -> Int32 = "lunaflux_process_inherited_wait"
