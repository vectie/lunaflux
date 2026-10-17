///|
#borrow(c)
extern "c" fn raw_graph_seal(
  c : NcclCommunicator,
  previous : UInt64,
  sites : Int,
) -> Int = "lunaflux_nccl_graph_seal"

///|
#borrow(c)
extern "c" fn raw_graph_begin(
  c : NcclCommunicator,
  generation : UInt64,
  plan : UInt64,
  first : UInt64,
) -> Int = "lunaflux_nccl_graph_begin"

///|
#borrow(c)
extern "c" fn raw_graph_complete(c : NcclCommunicator) -> Int = "lunaflux_nccl_graph_complete"

///|
#borrow(c)
extern "c" fn raw_graph_release(c : NcclCommunicator) -> Int = "lunaflux_nccl_graph_release"

///|
pub fn NcclCommunicator::seal_graph(
  self : Self,
  previous : UInt64,
  sites : Int,
) -> Unit raise NcclError {
  check_status(raw_graph_seal(self, previous, sites), CollectiveExecution)
}

///|
pub fn NcclCommunicator::begin_graph(
  self : Self,
  generation : UInt64,
  plan : UInt64,
  first : UInt64,
) -> Unit raise NcclError {
  check_status(
    raw_graph_begin(self, generation, plan, first),
    CollectiveExecution,
  )
}

///|
pub fn NcclCommunicator::complete_graph(self : Self) -> Bool raise NcclError {
  match raw_graph_complete(self) {
    0 => false
    1 => true
    status => {
      check_status(status, CollectiveExecution)
      false
    }
  }
}

///|
pub fn NcclCommunicator::release_graph(self : Self) -> Unit raise NcclError {
  check_status(raw_graph_release(self), CollectiveExecution)
}
