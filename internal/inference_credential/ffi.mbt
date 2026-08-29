///|
extern "c" fn raw_open_fixed() -> NativeInferenceCredential = "lunaflux_inference_credential_open_fixed"

///|
#borrow(native)
extern "c" fn raw_open_status(native : NativeInferenceCredential) -> Int = "lunaflux_inference_credential_open_status"

///|
#borrow(native, destination)
extern "c" fn raw_try_read(
  native : NativeInferenceCredential,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inference_credential_try_read"

///|
#borrow(native)
extern "c" fn raw_peek(native : NativeInferenceCredential) -> Int = "lunaflux_inference_credential_peek"

///|
#borrow(native)
extern "c" fn raw_close(native : NativeInferenceCredential) -> Int = "lunaflux_inference_credential_close"

///|
#borrow(native)
extern "c" fn raw_is_closed(native : NativeInferenceCredential) -> Bool = "lunaflux_inference_credential_is_closed"
