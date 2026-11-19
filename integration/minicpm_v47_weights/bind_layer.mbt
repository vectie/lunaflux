///|
fn bind_layer(
  indices : LayerIndices,
  buffers : ArrayView[@device.Allocation],
) -> @decoder.LayerWeights {
  let branch = match indices.branch {
    Recurrent(index) => {
      let weights : @recurrent.Weights = {
        qkv: buffers[index.qkv],
        gate: buffers[index.gate],
        decay: buffers[index.decay],
        beta: buffers[index.beta],
        output: buffers[index.output],
        convolution: buffers[index.convolution],
        norm: buffers[index.norm],
        a_log: buffers[index.a_log],
        dt_bias: buffers[index.dt_bias],
      }
      @decoder.BranchWeights::Recurrent(weights)
    }
    FullAttention(index) => {
      let weights : @attention.Weights = {
        query: buffers[index.query],
        key: buffers[index.key],
        value: buffers[index.value],
        output: buffers[index.output],
        query_norm: buffers[index.query_norm],
        key_norm: buffers[index.key_norm],
      }
      @decoder.BranchWeights::FullAttention(weights)
    }
  }
  let index = indices.moe
  let moe : @moe.Weights = {
    router: buffers[index.router],
    routed_bank: buffers[index.routed_bank],
    shared_gate: buffers[index.shared_gate],
    shared_up: buffers[index.shared_up],
    shared_down: buffers[index.shared_down],
    shared_gate_projection: buffers[index.shared_gate_projection],
  }
  {
    input_norm: buffers[indices.input_norm],
    post_attention_norm: buffers[indices.post_attention_norm],
    branch,
    moe,
  }
}
