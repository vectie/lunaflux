///|
fn add_flash_hyper(
  tensors : Array[Glm53TensorDescriptor],
  prefix : String,
) -> Unit {
  add_tensor(tensors, "\{prefix}.hc_attn_fn", BFloat16, [24L, 16384L])
  add_tensor(tensors, "\{prefix}.hc_attn_base", Float32, [24L])
  add_tensor(tensors, "\{prefix}.hc_attn_scale", Float32, [3L])
  add_tensor(tensors, "\{prefix}.hc_ffn_fn", BFloat16, [24L, 16384L])
  add_tensor(tensors, "\{prefix}.hc_ffn_base", Float32, [24L])
  add_tensor(tensors, "\{prefix}.hc_ffn_scale", Float32, [3L])
}

///|
fn add_flash_kda(
  tensors : Array[Glm53TensorDescriptor],
  prefix : String,
) -> Unit {
  add_tensor(tensors, "\{prefix}.A_log", Float32, [64L])
  add_tensor(tensors, "\{prefix}.dt_bias", Float32, [8192L])
  add_tensor(tensors, "\{prefix}.b_proj.weight", BFloat16, [64L, 4096L])
  add_tensor(tensors, "\{prefix}.f_a_proj.weight", BFloat16, [128L, 4096L])
  add_tensor(tensors, "\{prefix}.f_b_proj.weight", BFloat16, [8192L, 128L])
  add_tensor(tensors, "\{prefix}.g_a_proj.weight", BFloat16, [128L, 4096L])
  add_tensor(tensors, "\{prefix}.g_b_proj.weight", BFloat16, [8192L, 128L])
  for stem in ["q", "k", "v"] {
    add_tensor(tensors, "\{prefix}.\{stem}_proj.weight", BFloat16, [
      8192L, 4096L,
    ])
    add_tensor(tensors, "\{prefix}.\{stem}_conv1d.weight", BFloat16, [
      8192L, 1L, 4L,
    ])
  }
  add_tensor(tensors, "\{prefix}.o_norm.weight", BFloat16, [128L])
  add_tensor(tensors, "\{prefix}.o_proj.weight", BFloat16, [4096L, 8192L])
}

///|
fn add_flash_vision(tensors : Array[Glm53TensorDescriptor]) -> Unit {
  add_tensor(tensors, "model.visual.patch_embed.proj.weight", BFloat16, [
    1024L, 3L, 2L, 14L, 14L,
  ])
  add_tensor(tensors, "model.visual.patch_embed.proj.bias", BFloat16, [1024L])
  for layer in 0..<24 {
    let prefix = "model.visual.blocks.\{layer}"
    add_tensor(tensors, "\{prefix}.norm1.weight", BFloat16, [1024L])
    add_tensor(tensors, "\{prefix}.norm2.weight", BFloat16, [1024L])
    add_tensor(tensors, "\{prefix}.attn.qkv.weight", BFloat16, [3072L, 1024L])
    add_tensor(tensors, "\{prefix}.attn.qkv.bias", BFloat16, [3072L])
    add_tensor(tensors, "\{prefix}.attn.q_norm.weight", BFloat16, [64L])
    add_tensor(tensors, "\{prefix}.attn.k_norm.weight", BFloat16, [64L])
    add_tensor(tensors, "\{prefix}.attn.proj.weight", BFloat16, [1024L, 1024L])
    add_tensor(tensors, "\{prefix}.attn.proj.bias", BFloat16, [1024L])
    add_tensor(tensors, "\{prefix}.mlp.gate_proj.weight", BFloat16, [
      4096L, 1024L,
    ])
    add_tensor(tensors, "\{prefix}.mlp.gate_proj.bias", BFloat16, [4096L])
    add_tensor(tensors, "\{prefix}.mlp.up_proj.weight", BFloat16, [4096L, 1024L])
    add_tensor(tensors, "\{prefix}.mlp.up_proj.bias", BFloat16, [4096L])
    add_tensor(tensors, "\{prefix}.mlp.down_proj.weight", BFloat16, [
      1024L, 4096L,
    ])
    add_tensor(tensors, "\{prefix}.mlp.down_proj.bias", BFloat16, [1024L])
  }
  add_tensor(tensors, "model.visual.post_layernorm.weight", BFloat16, [1024L])
  add_tensor(tensors, "model.visual.downsample.weight", BFloat16, [
    4096L, 1024L, 2L, 2L,
  ])
  add_tensor(tensors, "model.visual.downsample.bias", BFloat16, [4096L])
  add_tensor(tensors, "model.visual.merger.gate_proj.weight", BFloat16, [
    10240L, 4096L,
  ])
  add_tensor(tensors, "model.visual.merger.up_proj.weight", BFloat16, [
    10240L, 4096L,
  ])
  add_tensor(tensors, "model.visual.merger.down_proj.weight", BFloat16, [
    4096L, 10240L,
  ])
  add_tensor(tensors, "model.visual.merger.proj.weight", BFloat16, [
    4096L, 4096L,
  ])
  add_tensor(
    tensors,
    "model.visual.merger.post_projection_norm.weight",
    BFloat16,
    [4096L],
  )
  add_tensor(
    tensors,
    "model.visual.merger.post_projection_norm.bias",
    BFloat16,
    [4096L],
  )
}
