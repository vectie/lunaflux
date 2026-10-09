///|
/// Exact installed checkpoint names and shapes. Returned order matches the
/// execution branch's named weights; rank-three convolution remains rank three
/// in the file and is only viewed as contiguous channel/lag rows by lowering.
pub fn bind_recurrent_weights(
  checkpoint : @precision_materialize.PackedCheckpoint,
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  guard spec.attention_kind(layer_index) is KimiDeltaAttention else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let hidden = spec.hidden_size()
  let heads = spec.linear_head_count()
  let dimension = spec.linear_head_dimension()
  let channels = heads * dimension
  let prefix = "model.language_model.layers.\{layer_index}.self_attn."
  let matrices = [
    ("q_proj.weight", channels, hidden),
    ("k_proj.weight", channels, hidden),
    ("v_proj.weight", channels, hidden),
    ("f_a_proj.weight", dimension, hidden),
    ("f_b_proj.weight", channels, dimension),
    ("b_proj.weight", heads, hidden),
  ]
  let weights = matrices.map(item => {
    let (name, rows, columns) = item
    checkpoint.bind_weight(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows~,
        columns~,
      ),
      payload_name=prefix + name,
    )
  })
  for (name, columns) in [("dt_bias", channels), ("A_log", heads)] {
    weights.push(
      checkpoint.bind_vector(
        representation=@precision_format.Representation::new(
          format=F32,
          rows=1,
          columns~,
        ),
        payload_name=prefix + name,
      ),
    )
  }
  for
    (name, rows, columns) in [
      ("g_a_proj.weight", dimension, hidden),
      ("g_b_proj.weight", channels, dimension),
    ] {
    weights.push(
      checkpoint.bind_weight(
        representation=@precision_format.Representation::new(
          format=Bf16,
          rows~,
          columns~,
        ),
        payload_name=prefix + name,
      ),
    )
  }
  weights.push(
    checkpoint.bind_vector(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=1,
        columns=dimension,
      ),
      payload_name=prefix + "o_norm.weight",
    ),
  )
  weights.push(
    checkpoint.bind_weight(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=hidden,
        columns=channels,
      ),
      payload_name=prefix + "o_proj.weight",
    ),
  )
  let kernel = spec.linear_conv_kernel_size()
  for stem in ["q", "k", "v"] {
    weights.push(
      checkpoint.bind_dense_tensor(
        representation=@precision_format.Representation::new(
          format=Bf16,
          rows=channels,
          columns=kernel,
        ),
        payload_name=prefix + stem + "_conv1d.weight",
        source_shape=[channels.to_int64(), 1L, kernel.to_int64()],
      ),
    )
  }
  weights
}

///|
fn recurrent_device_weights(
  weights : @upload.CheckpointDeviceWeights,
) -> @recurrent.RecurrentBlockWeights raise @device.DeviceError {
  {
    query: weights.allocation(0),
    key: weights.allocation(1),
    value: weights.allocation(2),
    forget_low: weights.allocation(3),
    forget_expansion: weights.allocation(4),
    beta: weights.allocation(5),
    dt_bias: weights.allocation(6),
    a_log: weights.allocation(7),
    gate_low: weights.allocation(8),
    gate_expansion: weights.allocation(9),
    norm: weights.allocation(10),
    output: weights.allocation(11),
    query_conv: weights.allocation(12),
    key_conv: weights.allocation(13),
    value_conv: weights.allocation(14),
  }
}
