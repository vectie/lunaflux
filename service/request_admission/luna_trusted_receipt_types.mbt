///|
const LUNA_TRUSTED_RECEIPT_IDLE : Int32 = 0

///|
const LUNA_TRUSTED_RECEIPT_LIVE : Int32 = 1

///|
pub struct LunaRequestReceiptWorkspace {
  priv clock_read : () -> UInt64 raise RequestAdmissionError
  priv budget_millis : UInt64
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut received_at_millis : UInt64
  priv mut deadline_at_millis : UInt64
}

///|
#valtype
pub struct LunaRequestReceipt {
  priv workspace : LunaRequestReceiptWorkspace
  priv epoch : UInt64
}
