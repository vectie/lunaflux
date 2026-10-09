///|
priv struct GlmDenseFfn {
  envelope : @hyper.HyperConnectionFrame
  mlp : @dense.DenseMlpFrame
  input_bytes : Int64
  mut input : @device.Allocation?
  launches : Array[@device.OrderedKernelLaunch]
}

///|
fn GlmDenseFfn::source_bytes(self : Self) -> Bytes {
  self.envelope.source_bytes() + b"\n" + self.mlp.source_bytes()
}

///|
fn GlmDenseFfn::workspace_bytes(self : Self) -> Int64 {
  self.input_bytes +
  self.envelope.workspace_bytes() +
  self.mlp.workspace_bytes()
}

///|
fn GlmDenseFfn::input(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.input is Some(input) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  input
}

///|
fn GlmDenseFfn::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  weights : @upload.CheckpointDeviceWeights,
  counts~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  let input = context.allocate(self.input_bytes)
  self.input = Some(input)
  self.envelope.prepare(
    context,
    aot,
    counts~,
    input~,
    output~,
    function_weight=weights.allocation(22),
    base=weights.allocation(23),
    scale=weights.allocation(24),
    norm_weight=weights.allocation(25),
    maximum_workspace_bytes=self.envelope.workspace_bytes(),
  )
  self.mlp.prepare(
    context,
    aot,
    counts~,
    input=self.envelope.branch_input(),
    gate=weights.allocation(19),
    up=weights.allocation(20),
    down=weights.allocation(21),
    output=self.envelope.branch_output(),
    maximum_workspace_bytes=self.mlp.workspace_bytes(),
  )
  for launch in self.envelope.prefix() {
    self.launches.push(launch)
  }
  for launch in self.mlp.launches() {
    self.launches.push(launch)
  }
  for launch in self.envelope.suffix() {
    self.launches.push(launch)
  }
}

///|
fn GlmDenseFfn::close(self : Self) -> Unit raise @device.DeviceError {
  self.launches.clear()
  self.mlp.close()
  self.envelope.close()
  if self.input is Some(input) {
    input.close()
    self.input = None
  }
}
