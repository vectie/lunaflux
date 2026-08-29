///|
extern "c" fn raw_inherited_wait(
  write_mode : Bool,
  timeout_millis : Int,
) -> Int = "lunaflux_process_inherited_wait"
