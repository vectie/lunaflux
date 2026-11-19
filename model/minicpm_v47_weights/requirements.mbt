///|
fn append(
  out : Array[Requirement],
  name : String,
  shape : Array[Int64],
  component : Component,
) -> Unit {
  out.push({ name, shape: ReadOnlyArray::from_array(shape), component, })
}

///|
fn affine(
  out : Array[Requirement],
  name : String,
  rows : Int64,
  columns : Int64,
  component : Component,
) -> Unit {
  append(out, name + ".weight", [rows, columns], component)
  append(out, name + ".bias", [rows], component)
}

///|
fn norm(
  out : Array[Requirement],
  name : String,
  width : Int64,
  component : Component,
) -> Unit {
  append(out, name + ".weight", [width], component)
  append(out, name + ".bias", [width], component)
}

///|
/// Complete tensor vocabulary for the ordinary BF16 decoder and vision path.
/// Routed experts remain packed [expert,out,in]; splitting or transposition is
/// a later explicit materialization choice. No MTP tensors occur in this source.
pub fn requirements(spec : @model.Spec) -> Array[Requirement] {
  let out = []
  let d = spec.decoder()
  let h = d.hidden_width().to_int64()
  let vocab = d.vocabulary_size().to_int64()
  append(out, "model.language_model.embed_tokens.weight", [vocab, h], Embedding)
  let f = d.full_attention()
  let r = d.recurrent_attention()
  let e = d.experts()
  let q_width = (f.query_heads * f.head_width).to_int64()
  let kv_width = (f.kv_heads * f.head_width).to_int64()
  let key_width = (r.key_heads * r.key_width).to_int64()
  let value_width = (r.value_heads * r.value_width).to_int64()
  for layer in 0..<d.layer_count() {
    let component = Decoder(layer)
    let prefix = "model.language_model.layers.\{layer}."
    for name in ["input_layernorm", "post_attention_layernorm"] {
      append(out, prefix + name + ".weight", [h], component)
    }
    if d.layer_kind(layer) is Some(FullAttention) {
      let a = prefix + "self_attn."
      append(out, a + "q_proj.weight", [q_width * 2L, h], component)
      for name in ["k_proj", "v_proj"] {
        append(out, a + name + ".weight", [kv_width, h], component)
      }
      append(out, a + "o_proj.weight", [h, q_width], component)
      for name in ["q_norm", "k_norm"] {
        append(out, a + name + ".weight", [f.head_width.to_int64()], component)
      }
    } else {
      let a = prefix + "linear_attn."
      for name in ["A_log", "dt_bias"] {
        append(out, a + name, [r.value_heads.to_int64()], component)
      }
      append(
        out,
        a + "conv1d.weight",
        [key_width * 2L + value_width, 1L, r.convolution_width.to_int64()],
        component,
      )
      for name in ["in_proj_a", "in_proj_b"] {
        append(
          out,
          a + name + ".weight",
          [r.value_heads.to_int64(), h],
          component,
        )
      }
      append(
        out,
        a + "in_proj_qkv.weight",
        [key_width * 2L + value_width, h],
        component,
      )
      append(out, a + "in_proj_z.weight", [value_width, h], component)
      append(out, a + "norm.weight", [r.value_width.to_int64()], component)
      append(out, a + "out_proj.weight", [h, value_width], component)
    }
    let m = prefix + "mlp."
    append(out, m + "gate.weight", [e.routed.to_int64(), h], component)
    append(
      out,
      m + "experts.gate_up_proj",
      [e.routed.to_int64(), 2L * e.intermediate_width.to_int64(), h],
      component,
    )
    append(
      out,
      m + "experts.down_proj",
      [e.routed.to_int64(), h, e.intermediate_width.to_int64()],
      component,
    )
    for name in ["gate_proj", "up_proj"] {
      append(
        out,
        m + "shared_expert." + name + ".weight",
        [e.shared_width.to_int64(), h],
        component,
      )
    }
    append(
      out,
      m + "shared_expert.down_proj.weight",
      [h, e.shared_width.to_int64()],
      component,
    )
    append(out, m + "shared_expert_gate.weight", [1L, h], component)
  }
  append(out, "model.language_model.norm.weight", [h], FinalNorm)
  append(out, "lm_head.weight", [vocab, h], LanguageHead)
  let v = spec.vision()
  let width = v.hidden_width.to_int64()
  let intermediate = v.intermediate_width.to_int64()
  let tower = "model.vision_tower."
  append(
    out,
    tower + "embeddings.patch_embedding.weight",
    [
      width,
      v.channels.to_int64(),
      v.patch_size.to_int64(),
      v.patch_size.to_int64(),
    ],
    VisionEmbedding,
  )
  append(
    out,
    tower + "embeddings.patch_embedding.bias",
    [width],
    VisionEmbedding,
  )
  append(
    out,
    tower + "embeddings.position_embedding.weight",
    [(v.position_grid_side * v.position_grid_side).to_int64(), width],
    VisionEmbedding,
  )
  for layer in 0..<v.layers {
    let component = VisionLayer(layer)
    let prefix = tower + "encoder.layers.\{layer}."
    for name in ["layer_norm1", "layer_norm2"] {
      norm(out, prefix + name, width, component)
    }
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      affine(out, prefix + "self_attn." + name, width, width, component)
    }
    affine(out, prefix + "mlp.fc1", intermediate, width, component)
    affine(out, prefix + "mlp.fc2", width, intermediate, component)
  }
  norm(out, tower + "post_layernorm", width, VisionFinalNorm)
  let patches = (v.window_side * v.window_side).to_int64()
  let merged = width * patches
  let early = tower + "vit_merger."
  norm(out, early + "layer_norm1", width, EarlyVisionMerger)
  norm(out, early + "pre_norm", merged, EarlyVisionMerger)
  for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    affine(out, early + "self_attn." + name, width, width, EarlyVisionMerger)
  }
  affine(
    out,
    early + "linear_1",
    intermediate * patches,
    merged,
    EarlyVisionMerger,
  )
  affine(
    out,
    early + "linear_2",
    width,
    intermediate * patches,
    EarlyVisionMerger,
  )
  let output_merger = "model.merger.mlp.0."
  norm(out, output_merger + "pre_norm", merged, OutputVisionMerger)
  affine(out, output_merger + "linear_1", merged, merged, OutputVisionMerger)
  affine(out, output_merger + "linear_2", h, merged, OutputVisionMerger)
  out
}
