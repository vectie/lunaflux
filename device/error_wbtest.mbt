///|
test "native failures translate without payload expansion" {
  let cases : Array[(@cuda.CudaError, DeviceError)] = [
    (Unavailable(ContextCreate), Unavailable(ContextCreate)),
    (InvalidArgument(CopyToDevice), InvalidArgument(CopyToDevice)),
    (Closed(EventRecord), Closed(EventRecord)),
    (Busy(ModuleClose), Busy(ModuleClose)),
    (InvalidArgument(AotKernelLaunch), InvalidArgument(AotKernelLaunch)),
    (DriverFailure(OrderedExecutorPoll), DriverFailure(OrderedExecutorPoll)),
    (DriverFailure(Bf16GemmRun), DriverFailure(Bf16GemmRun)),
    (HostAllocationFailed(ModuleLoad), HostAllocationFailed(ModuleLoad)),
    (SizeOverflow(AllocationCreate), SizeOverflow(AllocationCreate)),
    (
      InvalidArgument(AllocationRegionValidate),
      InvalidArgument(AllocationRegionValidate),
    ),
    (Unsupported(CublasLtCreate), Unsupported(CublasLtCreate)),
    (InvalidOutput(DeviceInventory), InvalidOutput(DeviceInventory)),
  ]
  for case in cases {
    assert_eq(error_from_cuda(case.0), case.1)
  }
}
