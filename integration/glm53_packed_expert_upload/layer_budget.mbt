///|
fn GlmRecurrentLayer::bound_weights(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let bound = bind_recurrent_weights(
    checkpoint,
    self.spec,
    layer_index=self.layer_index,
  )
  if self.hyper is Some(_) {
    bound.append(
      bind_attention_hyper_weights(checkpoint, self.spec, self.layer_index),
    )
  }
  if self.ffn is Some(ffn) {
    bound.append(
      match ffn.branch {
        Dense(_) =>
          bind_dense_ffn_weights(checkpoint, self.spec, self.layer_index)
        Moe(_) => {
          let weights = bind_router(checkpoint, layer_index=self.layer_index)
          weights.append(
            bind_ffn_hyper_weights(checkpoint, self.spec, self.layer_index),
          )
          weights
        }
      },
    )
  }
  bound
}

///|
fn GlmSparseLayer::bound_weights(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> (Array[@precision_materialize.PackedCheckpointWeight], Int) raise {
  let bound = @sparse_model.bind_flash_sparse_projection_weights(
    checkpoint,
    self.spec,
    layer_index=self.layer_index,
  )
  bound.append(
    @sparse_model.bind_flash_sparse_output_weights(
      checkpoint,
      self.spec,
      layer_index=self.layer_index,
    ),
  )
  bound.append(
    bind_attention_hyper_weights(checkpoint, self.spec, self.layer_index),
  )
  let ffn_offset = bound.length()
  bound.append(bind_router(checkpoint, layer_index=self.layer_index))
  bound.append(bind_ffn_hyper_weights(checkpoint, self.spec, self.layer_index))
  (bound, ffn_offset)
}

///|
fn GlmRecurrentLayer::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  let banks = match self.ffn {
    None => 0L
    Some(ffn) => ffn.bank_bytes()
  }
  self
  .bound_weights(checkpoint)
  .fold(init=self.workspace_bytes() + banks, (bytes, weight) => {
    bytes + weight.buffer_bytes()
  })
}

///|
fn GlmSparseLayer::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  let (bound, _) = self.bound_weights(checkpoint)
  bound.fold(
    init=self.workspace_bytes() + self.state_bytes() + self.ffn.bank_bytes(),
    (bytes, weight) => bytes + weight.buffer_bytes(),
  )
}
