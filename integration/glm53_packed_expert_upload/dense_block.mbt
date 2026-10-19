///|
/// Complete GLM early decoder block, including both residual-stream envelopes
/// and dense FFN, in the same recurrent queue. This constructor rejects MoE
/// layers rather than substituting dense computation for routed experts.
pub fn GlmRecurrentLayer::with_dense_block(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
  evaluation? : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  matrix_staging? : @matrix.MatrixStaging = Synchronous,
  hc_reduction? : @precision_ir.HyperControlReduction = Ordered,
  tensor_publication? : @precision_ir.ProjectionPublication = ReduceFloat32BeforeRound,
  tensor_group? : (Int, Int),
) -> Self raise {
  guard spec.feed_forward_kind(layer_index) is DenseSwiGlu else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let branch = GlmRecurrentLayer::with_hyper_connection(
    spec,
    layer_index~,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
    evaluation~,
    matrix_law~,
    matrix_staging~,
    hc_reduction~,
    tensor_publication~,
    tensor_group?,
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_streams(),
    hidden_width=spec.hidden_size(),
    maximum_rows~,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    mix_law=RoundedBf16ControlsAndProducts,
    control_square_law=hc_reduction.square_law(),
    control_dot_law=hc_reduction.dot_law(),
  )
  let mlp = @precision_ir.DenseSwiGluPrecision::new(
    dot_law=if matrix_law == ShapeAwareFloat32 {
      ShapeAwareFloat32
    } else {
      ScalarFloat32
    },
    hidden_width=spec.hidden_size(),
    intermediate_width=partition_width(
      spec.dense_intermediate_size(),
      tensor_group,
    ),
    maximum_rows~,
    publication=projection_publication(tensor_group, tensor_publication),
  )
  {
    spec,
    layer_index,
    program: branch.program,
    weights: branch.weights,
    hyper: branch.hyper,
    tensor_group,
    ffn: Some({
      envelope: @hyper.HyperConnectionFrame::new(
        @hyper_source.hyper_connection_source(
          envelope,
          symbol_prefix="glm_hyper_ffn_" + layer_index.to_string(),
        ),
      ),
      branch: Dense(
        @dense.DenseMlpFrame::new(
          @dense_source.dense_mlp_source(
            mlp,
            symbol_prefix="glm_dense_ffn_" + layer_index.to_string(),
          ),
        ),
      ),
      input_bytes: maximum_rows.to_int64() *
      spec.hyper_connection_streams().to_int64() *
      spec.hidden_size().to_int64() *
      2L,
      input: None,
      launches: [],
    }),
    status: Fresh,
  }
}

///|
fn bind_dense_ffn_weights(
  checkpoint : @precision_materialize.PackedCheckpoint,
  spec : @glm53_spec.Glm53FlashSpec,
  layer : Int,
  tensor_group? : (Int, Int),
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let prefix = "model.language_model.layers.\{layer}."
  let hidden = spec.hidden_size()
  let intermediate = spec.dense_intermediate_size()
  let weights : Array[@precision_materialize.PackedCheckpointWeight] = []
  for
    (name, rows, columns) in [
      ("mlp.gate_proj.weight", intermediate, hidden),
      ("mlp.up_proj.weight", intermediate, hidden),
      ("mlp.down_proj.weight", hidden, intermediate),
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
  if tensor_group is Some((rank, world_size)) {
    let (start, count) = @precision_ir.TensorAxisPartition::new(
      intermediate,
      rank~,
      world_size~,
    ).range()
    weights[0] = slice_weight_rows(weights[0], start, count)
    weights[1] = slice_weight_rows(weights[1], start, count)
    weights[2] = slice_weight_columns(weights[2], start, count)
  }
  weights.append(bind_ffn_hyper_weights(checkpoint, spec, layer))
  weights
}

///|
fn bind_ffn_hyper_weights(
  checkpoint : @precision_materialize.PackedCheckpoint,
  spec : @glm53_spec.Glm53FlashSpec,
  layer : Int,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let prefix = "model.language_model.layers.\{layer}."
  let hidden = spec.hidden_size()
  let streams = spec.hyper_connection_streams()
  let weights = [
    checkpoint.bind_weight(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=streams * (streams + 2),
        columns=streams * hidden,
      ),
      payload_name=prefix + "hc_ffn_fn",
    ),
  ]
  for
    (name, format, width) in [
      (
        "hc_ffn_base",
        @precision_format.Format::F32,
        spec.hyper_connection_streams() * (spec.hyper_connection_streams() + 2),
      ),
      ("hc_ffn_scale", F32, 3),
      ("post_attention_layernorm.weight", Bf16, hidden),
    ] {
    weights.push(
      checkpoint.bind_vector(
        representation=@precision_format.Representation::new(
          format~,
          rows=1,
          columns=width,
        ),
        payload_name=prefix + name,
      ),
    )
  }
  weights
}
