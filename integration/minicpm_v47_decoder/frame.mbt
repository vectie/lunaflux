///|
pub(all) struct Modules {
  envelope : @device.Module
  attention : @attention.Modules
  recurrent : @device.Module
  moe : @device.Module
}

///|
pub(all) struct Metadata {
  attention : @attention.Metadata
  request_slots : @device.Allocation
  reset : @device.Allocation
}

///|
pub(all) enum BranchWeights {
  Recurrent(@recurrent.Weights)
  FullAttention(@attention.Weights)
}

///|
pub(all) struct LayerWeights {
  input_norm : @device.Allocation
  post_attention_norm : @device.Allocation
  branch : BranchWeights
  moe : @moe.Weights
}

///|
/// Four separate caller-owned arenas; layer views follow Program::layers.
pub(all) struct StateArenas {
  convolution : @device.Allocation
  recurrent : @device.Allocation
  key_cache : @device.Allocation
  value_cache : @device.Allocation
}

///|
/// Separate mutable banks. MoE scratch retains its startup expert map and
/// cannot share storage with the branch arena or activation banks.
pub(all) struct Buffers {
  hidden : @device.Allocation
  normalized : @device.Allocation
  branch_output : @device.Allocation
  branch_workspace : @device.Allocation
  moe_workspace : @device.Allocation
}

///|
priv enum OwnedBranch {
  Recurrent(@recurrent.Frame)
  FullAttention(@attention.Frame)
}

///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns child frames, common functions and complete phase queues. Every
/// allocation/module is borrowed. Close executors before the frame, then owners.
pub struct Frame {
  priv program : Program
  priv functions : Array[@device.Function]
  priv branches : Array[OwnedBranch]
  priv experts : Array[@moe.Frame]
  priv prefill : Array[@device.OrderedKernelLaunch]
  priv decode : Array[@device.OrderedKernelLaunch]
  priv mut status : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  {
    program,
    functions: [],
    branches: [],
    experts: [],
    prefill: [],
    decode: [],
    status: Fresh,
  }
}

///|
pub fn Frame::launches(
  self : Self,
  phase : Phase,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match phase {
    Prefill => self.prefill[:]
    Decode => self.decode[:]
  }
}

///|
pub fn Frame::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.prefill.clear()
  self.decode.clear()
  while self.experts.last() is Some(frame) {
    frame.close()
    ignore(self.experts.pop())
  }
  while self.branches.last() is Some(frame) {
    match frame {
      Recurrent(frame) => frame.close()
      FullAttention(frame) => frame.close()
    }
    ignore(self.branches.pop())
  }
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  self.status = Closed
}
