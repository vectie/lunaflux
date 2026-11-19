///|
pub struct Plan {
  priv checkpoint : @checkpoint.PackedCheckpoint
  priv allocations : ReadOnlyArray[AllocationPlan]
  priv layers : ReadOnlyArray[LayerIndices]
  priv text : TextIndices
  priv bytes : Int64
}

///|
pub fn Plan::new(
  operators : @operators.DecoderOperators,
  checkpoint : @checkpoint.PackedCheckpoint,
) -> Self raise {
  let builder : Builder = { checkpoint, operators, entries: [], }
  let geometry = operators.geometry()
  let hidden = geometry.hidden_width()
  let full = geometry.full_attention()
  let linear = geometry.recurrent_attention()
  let experts = geometry.experts()
  let embedding = builder.matrix(
    "model.language_model.embed_tokens.weight",
    geometry.vocabulary_size(),
    hidden,
  )
  let layers = []
  for layer in 0..<geometry.layer_count() {
    let prefix = "model.language_model.layers.\{layer}."
    let input_norm = builder.vector(prefix + "input_layernorm.weight", hidden)
    let post_attention_norm = builder.vector(
      prefix + "post_attention_layernorm.weight",
      hidden,
    )
    let branch = match geometry.layer_kind(layer).unwrap() {
      GatedDeltaLinearAttention => {
        let a = prefix + "linear_attn."
        let channels = linear.key_heads * linear.key_width * 2 +
          linear.value_heads * linear.value_width
        BranchIndices::Recurrent({
          qkv: builder.matrix(a + "in_proj_qkv.weight", channels, hidden),
          gate: builder.matrix(
            a + "in_proj_z.weight",
            linear.value_heads * linear.value_width,
            hidden,
          ),
          decay: builder.matrix(
            a + "in_proj_a.weight",
            linear.value_heads,
            hidden,
          ),
          beta: builder.matrix(
            a + "in_proj_b.weight",
            linear.value_heads,
            hidden,
          ),
          output: builder.matrix(
            a + "out_proj.weight",
            hidden,
            linear.value_heads * linear.value_width,
          ),
          convolution: builder.tensor(a + "conv1d.weight", [
            channels.to_int64(),
            1L,
            linear.convolution_width.to_int64(),
          ]),
          norm: builder.vector(a + "norm.weight", linear.value_width),
          a_log: builder.vector(a + "A_log", linear.value_heads),
          dt_bias: builder.vector(a + "dt_bias", linear.value_heads),
        })
      }
      FullAttention => {
        let a = prefix + "self_attn."
        BranchIndices::FullAttention({
          query: builder.matrix(
            a + "q_proj.weight",
            full.query_heads * full.head_width * 2,
            hidden,
          ),
          key: builder.matrix(
            a + "k_proj.weight",
            full.kv_heads * full.head_width,
            hidden,
          ),
          value: builder.matrix(
            a + "v_proj.weight",
            full.kv_heads * full.head_width,
            hidden,
          ),
          output: builder.matrix(
            a + "o_proj.weight",
            hidden,
            full.query_heads * full.head_width,
          ),
          query_norm: builder.vector(a + "q_norm.weight", full.head_width),
          key_norm: builder.vector(a + "k_norm.weight", full.head_width),
        })
      }
    }
    let m = prefix + "mlp."
    let moe = {
      router: builder.matrix(m + "gate.weight", experts.routed, hidden),
      routed_bank: builder.experts(layer),
      shared_gate: builder.matrix(
        m + "shared_expert.gate_proj.weight",
        experts.shared_width,
        hidden,
      ),
      shared_up: builder.matrix(
        m + "shared_expert.up_proj.weight",
        experts.shared_width,
        hidden,
      ),
      shared_down: builder.matrix(
        m + "shared_expert.down_proj.weight",
        hidden,
        experts.shared_width,
      ),
      shared_gate_projection: builder.matrix(
        m + "shared_expert_gate.weight",
        1,
        hidden,
      ),
    }
    layers.push({ input_norm, post_attention_norm, branch, moe, })
  }
  let final_norm = builder.vector("model.language_model.norm.weight", hidden)
  let language_head = builder.matrix(
    "lm_head.weight",
    geometry.vocabulary_size(),
    hidden,
  )
  let mut bytes = 0L
  for entry in builder.entries {
    guard bytes <= 9223372036854775807L - entry.bytes else {
      fail("text weight budget overflow")
    }
    bytes += entry.bytes
  }
  {
    checkpoint,
    allocations: ReadOnlyArray::from_array(builder.entries),
    layers: ReadOnlyArray::from_array(layers),
    text: { embedding, final_norm, language_head, },
    bytes,
  }
}

///|
pub fn Plan::allocations(self : Self) -> ReadOnlyArray[AllocationPlan] {
  self.allocations
}

///|
pub fn Plan::layers(self : Self) -> ReadOnlyArray[LayerIndices] {
  self.layers
}

///|
pub fn Plan::text(self : Self) -> TextIndices {
  self.text
}

///|
pub fn Plan::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
/// Startup-only synchronous byte sink. The scratch view must be consumed before
/// returning; a single scoped session reuses shard handles across allocations.
pub fn Plan::transfer(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  consume~ : (Int32, Int64, FixedArray[Byte], Int32) -> Bool,
) -> Unit raise {
  guard scratch.length() > 0 else { fail("empty text weight scratch") }
  self.checkpoint.with_transfer_session(root, limits, checkpoint => {
    for ordinal, entry in self.allocations {
      checkpoint.transfer_weights_to(
        root,
        entry.weights,
        destinations=entry.destinations,
        destination_bytes=[entry.bytes],
        scratch~,
        limits~,
        consume=(_, offset, bytes, count) => {
          consume(ordinal, offset, bytes, count)
        },
      )
    }
  })
}
