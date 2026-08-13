///|
#borrow(output)
extern "c" fn raw_now_millis(output : Ref[UInt64]) -> Int = "lunaflux_monotonic_clock_now_millis"
