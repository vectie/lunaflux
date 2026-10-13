///|
fn GlmFfn::is_tensor_parallel(self : Self) -> Bool {
  self.branch is Moe(moe) && moe.tensor_partition is Some(_)
}

///|
fn GlmFfn::tensor_width(self : Self) -> Int? {
  match self.branch {
    Moe(moe) => moe.tensor_partition.map(p => p.collective_width())
    Dense(_) => None
  }
}

///|
fn GlmFfn::tensor_boundary(
  self : Self,
) -> (Int, @device.Allocation, Int)? raise @device.DeviceError {
  if self.branch is Moe(moe) && moe.tensor_boundary is Some(boundary) {
    Some(
      (
        self.envelope.prefix().length() + boundary.before_kernel(),
        boundary.partials(),
        boundary.width(),
      ),
    )
  } else {
    None
  }
}

///|
fn GlmDecoderLayer::tensor_width(self : Self) -> Int? {
  match self {
    Recurrent(layer) =>
      match layer.ffn {
        Some(ffn) => ffn.tensor_width()
        None => None
      }
    Sparse(layer) => layer.ffn.tensor_width()
  }
}

///|
fn GlmDecoderLayer::tensor_boundary(
  self : Self,
) -> (Int, @device.Allocation, Int)? raise @device.DeviceError {
  let ffn = match self {
    Recurrent(layer) => layer.ffn
    Sparse(layer) => Some(layer.ffn)
  }
  if ffn is Some(ffn) && ffn.tensor_boundary() is Some((cut, buffer, width)) {
    Some(
      (self.launches().length() - ffn.launches.length() + cut, buffer, width),
    )
  } else {
    None
  }
}

///|
/// Startup-only device contract. Expert slots are reduced before BF16
/// publication; shared experts and attention are replicated, not sharded.
pub fn GlmDecoderSlice::tensor_contract(
  self : Self,
  label : @device.TensorCollectiveContractDigest,
) -> @device.TensorCollectiveContract raise @device.DeviceError {
  guard self.tensor_group is Some((rank, world_size)) else {
    raise @device.DeviceError::InvalidArgument(TensorCollectiveContractCreate)
  }
  let sites = []
  for i, layer in self.layers {
    if layer.tensor_width() is Some(width) {
      sites.push(
        @device.TensorCollectiveSite::new(
          sequence=sites.length() + 1,
          operation_id=self.first_layer + i,
          kind=F32SumAllReduce,
          full_width=width,
          local_width=width,
        ),
      )
    }
  }
  @device.TensorCollectiveContract::new(
    digest=label,
    rank~,
    world_size~,
    max_query_tokens=self.maximum_rows,
    sites~,
  )
}

///|
pub fn GlmDecoderStage::tensor_contract(
  self : Self,
  label : @device.TensorCollectiveContractDigest,
) -> @device.TensorCollectiveContract raise @device.DeviceError {
  self.decoder.tensor_contract(label)
}
