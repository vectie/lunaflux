///|
priv enum GlmFfnBranch {
  Dense(@dense.DenseMlpFrame)
  Moe(GlmMoeBranch)
}

///|
priv struct GlmFfn {
  envelope : @hyper.HyperConnectionFrame
  branch : GlmFfnBranch
  input_bytes : Int64
  mut input : @device.Allocation?
  launches : Array[@device.OrderedKernelLaunch]
}

///|
fn GlmFfn::source_bytes(self : Self) -> Bytes {
  let branch = match self.branch {
    Dense(mlp) => mlp.source_bytes()
    Moe(moe) => moe.source
  }
  self.envelope.source_bytes() + b"\n" + branch
}

///|
fn GlmFfn::workspace_bytes(self : Self) -> Int64 {
  let branch = match self.branch {
    Dense(mlp) => mlp.workspace_bytes()
    Moe(moe) => moe.program.workspace_bytes()
  }
  self.input_bytes + self.envelope.workspace_bytes() + branch
}

///|
fn GlmFfn::bank_bytes(self : Self) -> Int64 {
  match self.branch {
    Dense(_) => 0L
    Moe(moe) => moe.routed.bank_bytes() + moe.shared.bank_bytes()
  }
}

///|
fn GlmFfn::owned_bank_bytes(self : Self) -> Int64 {
  match self.branch {
    Dense(_) => 0L
    Moe(moe) => {
      let routed = if moe.banks.length() > 0 {
        moe.routed.bank_bytes()
      } else {
        0L
      }
      let shared = if moe.banks.length() > 1 {
        moe.shared.bank_bytes()
      } else {
        0L
      }
      routed + shared
    }
  }
}

///|
fn GlmFfn::input(self : Self) -> @device.Allocation raise @device.DeviceError {
  guard self.input is Some(input) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  input
}

///|
fn GlmFfn::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  weights : @upload.CheckpointDeviceWeights,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  let input = context.allocate(self.input_bytes)
  self.input = Some(input)
  let base = match self.branch {
    Dense(_) => 22
    Moe(_) => 21
  }
  self.envelope.prepare(
    context,
    aot,
    counts~,
    input~,
    output~,
    function_weight=weights.allocation(base),
    base=weights.allocation(base + 1),
    scale=weights.allocation(base + 2),
    norm_weight=weights.allocation(base + 3),
    maximum_workspace_bytes=self.envelope.workspace_bytes(),
  )
  let branch = match self.branch {
    Dense(mlp) => {
      mlp.prepare(
        context,
        aot,
        counts~,
        input=self.envelope.branch_input(),
        gate=weights.allocation(19),
        up=weights.allocation(20),
        down=weights.allocation(21),
        output=self.envelope.branch_output(),
        maximum_workspace_bytes=mlp.workspace_bytes(),
      )
      mlp.launches()
    }
    Moe(moe) => {
      moe.prepare(
        context,
        aot,
        weights,
        checkpoint,
        root,
        layer_index~,
        chunk_bytes~,
        limits~,
        counts~,
        input=self.envelope.branch_input(),
        output=self.envelope.branch_output(),
      )
      moe.program.launches()
    }
  }
  for launch in self.envelope.prefix() {
    self.launches.push(launch)
  }
  for launch in branch {
    self.launches.push(launch)
  }
  for launch in self.envelope.suffix() {
    self.launches.push(launch)
  }
}

///|
fn GlmFfn::close(self : Self) -> Unit raise @device.DeviceError {
  self.launches.clear()
  match self.branch {
    Dense(mlp) => mlp.close()
    Moe(moe) => moe.close()
  }
  self.envelope.close()
  if self.input is Some(input) {
    input.close()
    self.input = None
  }
}
