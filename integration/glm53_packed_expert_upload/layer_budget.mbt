///|
fn GlmRecurrentLayer::bound_weights(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let bound = bind_recurrent_weights(
    checkpoint,
    self.spec,
    layer_index=self.layer_index,
    tensor_group?=self.tensor_group,
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
          bind_dense_ffn_weights(
            checkpoint,
            self.spec,
            self.layer_index,
            tensor_group?=self.tensor_group,
          )
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
  if self.tensor_group is Some((rank, world_size)) {
    let (qfirst, qcount) = @precision_ir.TensorAxisPartition::new(
      self.spec.attention_head_count(),
      rank~,
      world_size~,
    ).range()
    let (kfirst, kcount) = @precision_ir.TensorAxisPartition::new(
      self.spec.kv_head_count(),
      rank~,
      world_size~,
    ).range()
    let qdim = self.spec.qk_nope_head_dimension()
    let vdim = self.spec.value_head_dimension()
    bound[2] = slice_weight_rows(bound[2], qfirst * qdim, qcount * qdim)
    bound[5] = slice_weight_rows(
      bound[5],
      kfirst * (qdim + vdim),
      kcount * (qdim + vdim),
    )
    bound[12] = slice_weight_columns(bound[12], qfirst * vdim, qcount * vdim)
  }
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
