///|
extern "c" fn raw_open_fixed() -> NativePromotionVerifierKey = "lunaflux_promotion_verifier_key_open_fixed"

///|
#borrow(native)
extern "c" fn raw_open_status(native : NativePromotionVerifierKey) -> Int = "lunaflux_promotion_verifier_key_open_status"

///|
#borrow(native, destination)
extern "c" fn raw_try_read(
  native : NativePromotionVerifierKey,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_promotion_verifier_key_try_read"

///|
#borrow(native)
extern "c" fn raw_peek(native : NativePromotionVerifierKey) -> Int = "lunaflux_promotion_verifier_key_peek"

///|
#borrow(native)
extern "c" fn raw_close(native : NativePromotionVerifierKey) -> Int = "lunaflux_promotion_verifier_key_close"

///|
#borrow(native)
extern "c" fn raw_is_closed(native : NativePromotionVerifierKey) -> Bool = "lunaflux_promotion_verifier_key_is_closed"
