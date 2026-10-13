///|
#borrow(c)
extern "c" fn raw_prepare_deferred(c : NcclCommunicator, capacity : Int) -> Int = "lunaflux_nccl_prepare_deferred"

///|
#borrow(c)
extern "c" fn raw_defer_submitted(c : NcclCommunicator) -> Int = "lunaflux_nccl_defer_submitted"

///|
#borrow(c)
extern "c" fn raw_poll_deferred(c : NcclCommunicator) -> Int = "lunaflux_nccl_poll_deferred"

///|
pub fn NcclCommunicator::prepare_deferred(
  self : Self,
  capacity : Int,
) -> Unit raise NcclError {
  check_status(raw_prepare_deferred(self, capacity), CollectiveExecution)
}

///|
pub fn NcclCommunicator::defer_submitted(self : Self) -> Bool raise NcclError {
  match raw_defer_submitted(self) {
    0 => false
    1 => true
    status => {
      check_status(status, CollectiveExecution)
      false
    }
  }
}

///|
pub fn NcclCommunicator::poll_deferred(self : Self) -> Bool raise NcclError {
  match raw_poll_deferred(self) {
    0 => false
    1 => true
    status => {
      check_status(status, CollectiveExecution)
      false
    }
  }
}
