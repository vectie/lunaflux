///|
extern "c" fn raw_open_fixed() -> NativeInheritedDrain = "lunaflux_inherited_drain_open_fixed"

///|
#borrow(native)
extern "c" fn raw_open_status(native : NativeInheritedDrain) -> Int = "lunaflux_inherited_drain_open_status"

///|
#borrow(native, destination)
extern "c" fn raw_try_read(
  native : NativeInheritedDrain,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inherited_drain_try_read"

///|
#borrow(native)
extern "c" fn raw_peek(native : NativeInheritedDrain) -> Int = "lunaflux_inherited_drain_peek"

///|
#borrow(native, source)
extern "c" fn raw_try_write(
  native : NativeInheritedDrain,
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inherited_drain_try_write"

///|
#borrow(native)
extern "c" fn raw_close(native : NativeInheritedDrain) -> Int = "lunaflux_inherited_drain_close"

///|
#borrow(native)
extern "c" fn raw_is_closed(native : NativeInheritedDrain) -> Bool = "lunaflux_inherited_drain_is_closed"
