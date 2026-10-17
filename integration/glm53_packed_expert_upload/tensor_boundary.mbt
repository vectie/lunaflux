///|
fn GlmDecoderLayer::producer_fork(
  self : Self,
) -> @parallel.ForkJoinSpan? raise @device.DeviceError {
  let ffn = match self {
    Recurrent(layer) => layer.ffn
    Sparse(layer) => Some(layer.ffn)
  }
  if ffn is Some(ffn) &&
    ffn.branch is Moe(moe) &&
    moe.program.producer_fork() is Some(span) {
    Some(
      span.shift(
        self.launches().length() -
        ffn.launches.length() +
        ffn.envelope.prefix().length(),
      ) catch {
        _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
      },
    )
  } else {
    None
  }
}

///|
fn GlmFfn::is_tensor_parallel(self : Self) -> Bool {
  self.tensor_width() is Some(_)
}

///|
fn GlmFfn::tensor_width(self : Self) -> Int? {
  match self.branch {
    Moe(moe) =>
      match moe.program.coalesced_tensor_width() {
        Some(width) => Some(width)
        None => moe.tensor_partition.map(p => p.collective_width())
      }
    Dense(mlp) => mlp.tensor_width()
  }
}

///|
fn GlmFfn::tensor_boundary(
  self : Self,
) -> (Int, @device.Allocation, Int)? raise @device.DeviceError {
  if self.branch is Dense(mlp) &&
    mlp.tensor_boundary() is Some((cut, buffer, width)) {
    return Some((self.envelope.prefix().length() + cut, buffer, width))
  }
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
fn GlmDecoderLayer::tensor_widths(self : Self) -> Array[(Int, Int)] {
  let widths = []
  let ffn = match self {
    Recurrent(layer) => {
      if layer.program.tensor_width() is Some(width) {
        widths.push((0, width))
      }
      layer.ffn
    }
    Sparse(layer) => {
      if layer.decoder.tensor_width() is Some(width) {
        widths.push((0, width))
      }
      Some(layer.ffn)
    }
  }
  if ffn is Some(ffn) && ffn.tensor_width() is Some(width) {
    widths.push((1, width))
    if ffn.branch is Moe(moe) &&
      moe.shared_partition is Some(partition) &&
      moe.program.coalesced_tensor_width() is None {
      widths.push((2, partition.collective_width()))
    }
  }
  widths
}

///|
fn GlmDecoderLayer::tensor_boundaries(
  self : Self,
) -> Array[(Int, Int, @device.Allocation, Int)] raise @device.DeviceError {
  let boundaries = []
  let ffn = match self {
    Recurrent(layer) => {
      if layer.program.tensor_boundary() is Some((cut, buffer, width)) {
        boundaries.push((0, cut, buffer, width))
      }
      layer.ffn
    }
    Sparse(layer) => {
      if layer.decoder.tensor_boundary() is Some((cut, buffer, width)) {
        boundaries.push((0, cut, buffer, width))
      }
      Some(layer.ffn)
    }
  }
  if ffn is Some(ffn) && ffn.tensor_boundary() is Some((cut, buffer, width)) {
    boundaries.push(
      (1, self.launches().length() - ffn.launches.length() + cut, buffer, width),
    )
    if ffn.branch is Moe(moe) &&
      moe.program.shared_tensor_boundary() is Some(site) {
      boundaries.push(
        (
          2,
          self.launches().length() -
          ffn.launches.length() +
          ffn.envelope.prefix().length() +
          site.before_kernel(),
          site.partials(),
          site.width(),
        ),
      )
    }
  }
  boundaries
}

///|
/// Startup-only ordered effects. Each sublayer publishes once after reduction.
pub fn GlmDecoderSlice::tensor_contract(
  self : Self,
  label : @device.TensorCollectiveContractDigest,
) -> @device.TensorCollectiveContract raise @device.DeviceError {
  guard self.tensor_group is Some((rank, world_size)) else {
    raise @device.DeviceError::InvalidArgument(TensorCollectiveContractCreate)
  }
  let sites = []
  for i, layer in self.layers {
    for (sublayer, width) in layer.tensor_widths() {
      sites.push(
        @device.TensorCollectiveSite::new(
          sequence=sites.length() + 1,
          operation_id=(self.first_layer + i) * 3 + sublayer,
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
