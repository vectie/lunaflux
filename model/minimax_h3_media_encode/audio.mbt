///|
pub struct AudioEncoder {
  graph : @plan.Graph
  samples : Int
  latent_time : Int
  means_value : Int
  deviations_value : Int
}

///|
priv struct Builder {
  video : Bool
  weight_ids : Map[String, Int]
  values : Array[@plan.Value]
  nodes : Array[@plan.Node]
}

///|
fn Builder::weight(
  self : Self,
  name : String,
  shape : Array[Int64],
) -> Int raise @plan.PlanError {
  let matches = if self.video {
    @weights.MiniMaxH3VideoVaeSchema::official_v1().matches_tensor(
      name,
      Float32,
      ReadOnlyArray::from_array(shape),
    )
  } else {
    @weights.MiniMaxH3AudioVaeSchema::official_v1().matches_tensor(
      name,
      Float32,
      ReadOnlyArray::from_array(shape),
    )
  }
  guard matches else { raise @plan.PlanError::InvalidGeometry }
  if self.weight_ids.get(name) is Some(id) {
    return id
  }
  let elements = shape.fold(init=1L, (a, b) => a * b)
  let id = self.values.length()
  self.values.push({ elements, origin: Weight(name), })
  self.weight_ids[name] = id
  id
}

///|
fn Builder::emit(
  self : Self,
  operation : @plan.Operation,
  inputs : Array[Int],
) -> Int raise @plan.PlanError {
  let (_, elements) = operation.signature()
  let output = self.values.length()
  self.values.push({ elements, origin: Temporary, })
  self.nodes.push({
    operation,
    inputs: ReadOnlyArray::from_array(inputs),
    output,
  })
  output
}

///|
fn Builder::conv(
  self : Self,
  input : Int,
  name : String,
  ci : Int,
  co : Int,
  time : Int,
  kernel : Int,
  dilation : Int,
  stride : Int,
  padding : Int,
) -> (Int, Int) raise @plan.PlanError {
  let g = self.weight(name + ".weight_g", [co.to_int64(), 1L, 1L])
  let v = self.weight(name + ".weight_v", [
    co.to_int64(),
    ci.to_int64(),
    kernel.to_int64(),
  ])
  let bias = self.weight(name + ".bias", [co.to_int64()])
  let normalized = self.emit(WeightNorm(outputs=co, per_output=ci * kernel), [
    g, v,
  ])
  let output_time = (time + 2 * padding - dilation * (kernel - 1) - 1) / stride +
    1
  (
    self.emit(
      Conv(
        batch=2,
        input_channels=ci,
        output_channels=co,
        input_time=time,
        output_time~,
        kernel~,
        dilation~,
        stride~,
        padding~,
      ),
      [input, normalized, bias],
    ),
    output_time,
  )
}

///|
fn Builder::snake(
  self : Self,
  input : Int,
  name : String,
  channels : Int,
  time : Int,
) -> Int raise @plan.PlanError {
  self.emit(Snake(batch=2, channels~, time~), [
    input,
    self.weight(name + ".alpha", [1L, channels.to_int64(), 1L]),
  ])
}

///|
fn Builder::norm(
  self : Self,
  input : Int,
  name : String,
  channels : Int,
  time : Int,
) -> Int raise @plan.PlanError {
  self.emit(LayerNorm(batch=2, channels~, time~), [
    input,
    self.weight(name + ".weight", [channels.to_int64()]),
    self.weight(name + ".bias", [channels.to_int64()]),
  ])
}

///|
fn Builder::dense(
  self : Self,
  input : Int,
  name : String,
  ci : Int,
  co : Int,
  time : Int,
  pointwise? : Bool = false,
) -> Int raise @plan.PlanError {
  let shape = if pointwise {
    [co.to_int64(), ci.to_int64(), 1L]
  } else {
    [co.to_int64(), ci.to_int64()]
  }
  self.emit(Dense(batch=2, input_channels=ci, output_channels=co, time~), [
    input,
    self.weight(name + ".weight", shape),
    self.weight(name + ".bias", [co.to_int64()]),
  ])
}

///|
/// Exact deterministic mean encoder for prepared stereo 32 kHz F32 PCM.
/// Resampling/media decoding is an ingress responsibility. Unlike video VAE
/// encoding, audio uses mean_proj, never logs_proj or a random posterior draw.
pub fn AudioEncoder::new(
  samples~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise @plan.PlanError {
  guard samples > 0 && samples <= 2147482847 else {
    raise @plan.PlanError::InvalidGeometry
  }
  let padded = (samples + 799) / 800 * 800
  let b : Builder = {
    video: false,
    weight_ids: Map([]),
    values: [{ elements: samples.to_int64() * 2L, origin: Input, }],
    nodes: [],
  }
  let input = b.emit(Pad(batch=2, input_time=samples, output_time=padded), [0])
  let (first, _) = b.conv(input, "encoder.block.0", 1, 64, padded, 7, 1, 1, 3)
  let mut state = first
  let mut channels = 64
  let mut time = padded
  for stage, rate in [2, 4, 4, 5, 5] {
    let prefix = "encoder.block.\{stage + 1}.block"
    for unit, dilation in [1, 3, 9] {
      let name = "\{prefix}.\{unit}.block"
      let a = b.snake(state, name + ".0", channels, time)
      let (c, _) = b.conv(
        a,
        name + ".1",
        channels,
        channels,
        time,
        7,
        dilation,
        1,
        3 * dilation,
      )
      let a = b.snake(c, name + ".2", channels, time)
      let (c, _) = b.conv(a, name + ".3", channels, channels, time, 1, 1, 1, 0)
      state = b.emit(Add(elements=2L * channels.to_int64() * time.to_int64()), [
        state, c,
      ])
    }
    let a = b.snake(state, prefix + ".3", channels, time)
    let (next, next_time) = b.conv(
      a,
      prefix + ".4",
      channels,
      channels * 2,
      time,
      2 * rate,
      1,
      rate,
      (rate + 1) / 2,
    )
    state = next
    channels *= 2
    time = next_time
  }
  state = b.snake(state, "encoder.block.6", 2048, time)
  let (encoded, _) = b.conv(
    state, "encoder.block.7", 2048, 2048, time, 3, 1, 1, 1,
  )
  let norm = b.norm(encoded, "pre_block.norm1", 2048, time)
  let qkv = b.emit(Qkv(batch=2, channels=2048, time~), [
    norm,
    b.weight("pre_block.attn.qkv.weight", [6144L, 2048L]),
    b.weight("pre_block.attn.q_bias", [2048L]),
    b.weight("pre_block.attn.zero_k_bias", [2048L]),
    b.weight("pre_block.attn.v_bias", [2048L]),
  ])
  let attention = b.emit(
    CausalAttention(batch=2, heads=8, head_dim=256, time~),
    [qkv],
  )
  let pooled = b.emit(
    MeanHeadsPool(batch=2, heads=8, head_dim=256, output_channels=32, time~),
    [attention],
  )
  let attention = b.dense(pooled, "pre_block.attn.proj", 32, 32, time)
  let norm = b.norm(encoded, "pre_block.norm3", 2048, time)
  let residual = b.dense(norm, "pre_block.proj", 2048, 32, time)
  let joined = b.emit(Add(elements=64L * time.to_int64()), [residual, attention])
  let norm = b.norm(joined, "pre_block.norm2", 32, time)
  let norm = b.norm(norm, "pre_block.mlp.norm", 32, time)
  let gate = b.dense(norm, "pre_block.mlp.w0", 32, 64, time)
  let value = b.dense(norm, "pre_block.mlp.w1", 32, 64, time)
  let product = b.emit(GeluMultiply(elements=128L * time.to_int64()), [
    gate, value,
  ])
  let branch = b.dense(product, "pre_block.mlp.w2", 64, 32, time)
  let hidden = b.emit(Add(elements=64L * time.to_int64()), [joined, branch])
  let latent = b.dense(hidden, "mean_proj", 32, 32, time, pointwise=true)
  let means_value = b.values.length()
  b.values.push({ elements: 32L, origin: Input, })
  let deviations_value = b.values.length()
  b.values.push({ elements: 32L, origin: Input, })
  let output = b.emit(NormalizeRows(batch=2, channels=32, time~), [
    latent, means_value, deviations_value,
  ])
  {
    graph: @plan.Graph::new(b.values, b.nodes, output~, max_scratch_bytes~),
    samples,
    latent_time: time,
    means_value,
    deviations_value,
  }
}
