///|
#external
type RawSession

///|
#external
type RawTensor

///|
#borrow(path)
extern "c" fn session_open(
  path : Bytes,
  device : Int,
  budget : Int64,
) -> RawSession = "lfr_bridge_open"

///|
extern "c" fn session_valid(session : RawSession) -> Bool = "lfr_bridge_valid"

///|
extern "c" fn session_same(a : RawSession, b : RawSession) -> Bool = "lfr_bridge_same"

///|
extern "c" fn session_close(session : RawSession) -> Unit = "lfr_bridge_close"

///|
extern "c" fn session_error(session : RawSession) -> Bytes = "lfr_bridge_error"

///|
extern "c" fn tensor_null() -> RawTensor = "lfr_bridge_null"

///|
extern "c" fn tensor_valid(tensor : RawTensor) -> Bool = "lfr_bridge_tensor_valid"

///|
extern "c" fn tensor_close(session : RawSession, tensor : RawTensor) -> Unit = "lfr_bridge_tensor_close"

///|
extern "c" fn live_bytes(session : RawSession) -> Int64 = "lfr_bridge_live_bytes"

///|
#borrow(path)
extern "c" fn read_file(
  session : RawSession,
  path : Bytes,
  offset : Int64,
  length : Int,
) -> Bytes = "lfr_bridge_read"

///|
#borrow(path)
extern "c" fn load_bf16(
  session : RawSession,
  path : Bytes,
  offset : Int64,
  rows : Int,
  columns : Int,
) -> RawTensor = "lfr_bridge_load"

///|
extern "c" fn apply(
  session : RawSession,
  op : Int,
  x : RawTensor,
  y : RawTensor,
  z : RawTensor,
  a : Int,
  b : Int,
  c : Int,
  d : Double,
) -> RawTensor = "lfr_bridge_apply"

///|
#borrow(path)
extern "c" fn write_bf16(
  session : RawSession,
  tensor : RawTensor,
  path : Bytes,
) -> Int = "lfr_bridge_write"
