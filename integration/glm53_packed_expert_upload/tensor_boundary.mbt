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
fn GlmFfn::tensor_width(self : Self) -> Int32? {
  match self.branch {
    Moe(moe) =>
      match moe.program.combined_tensor_width() {
        Some(width) => Some(width)
        None => moe.tensor_partition.map(p => p.collective_width())
      }
    Dense(mlp) => mlp.tensor_width()
  }
}

///|
fn GlmFfn::tensor_boundary(
  self : Self,
) -> (Int32, @device.Allocation, Int32)? raise @device.DeviceError {
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
fn GlmDecoderLayer::tensor_widths(self : Self) -> Array[(Int32, Int32)] {
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
      moe.program.combined_tensor_width() is None {
      widths.push((2, partition.collective_width()))
    }
  }
  widths
}

///|
fn GlmDecoderLayer::tensor_boundaries(
  self : Self,
) -> Array[(Int32, Int32, @device.Allocation, Int32)] raise @device.DeviceError {
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
  vocabulary_shard? : @parallel.VocabularyShard,
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
          kind=layer.tensor_kind(sublayer),
          full_width=width,
          local_width=width,
        ),
      )
    }
  }
  if vocabulary_shard is Some(shard) {
    guard shard.group() == (rank, world_size) else {
      raise @device.DeviceError::InvalidArgument(TensorCollectiveContractCreate)
    }
    sites.push(
      @device.TensorCollectiveSite::new(
        sequence=sites.length() + 1,
        operation_id=self.end_layer * 3,
        kind=F32AllGather,
        full_width=shard.transfer_width() * world_size,
        local_width=shard.transfer_width(),
      ),
    )
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
  self.decoder.tensor_contract(
    label,
    vocabulary_shard?=self.text.vocabulary_shard,
  )
}

///|
fn GlmDecoderLayer::tensor_kind(
  self : Self,
  sublayer : Int32,
) -> @device.TensorCollectiveKind {
  let ffn = match self {
    Recurrent(layer) => layer.ffn
    Sparse(layer) => Some(layer.ffn)
  }
  if sublayer == 1 && ffn is Some(ffn) && ffn.branch is Moe(moe) {
    moe.program.tensor_kind()
  } else {
    F32SumAllReduce
  }
}
