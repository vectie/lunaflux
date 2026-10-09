///|
fn append_audio_weight_norm(names : Array[String], prefix : String) -> Unit {
  names.push("\{prefix}.bias")
  names.push("\{prefix}.weight_g")
  names.push("\{prefix}.weight_v")
}

///|
fn append_audio_activation(names : Array[String], prefix : String) -> Unit {
  names.push("\{prefix}.act.alpha")
  names.push("\{prefix}.act.beta")
  names.push("\{prefix}.downsample.lowpass.filter")
  names.push("\{prefix}.upsample.filter")
}

///|
fn expected_audio_vae_names() -> Array[String] {
  let names : Array[String] = []
  append_weight_bias(names, "dec_in_proj")
  append_audio_activation(names, "decoder.activation_post")
  names.push("decoder.conv_post.weight_g")
  names.push("decoder.conv_post.weight_v")
  append_audio_weight_norm(names, "decoder.conv_pre")
  for block in 0..<21 {
    let prefix = "decoder.resblocks.\{block}"
    for activation in 0..<6 {
      append_audio_activation(names, "\{prefix}.activations.\{activation}")
    }
    for convolution in 0..<3 {
      append_audio_weight_norm(names, "\{prefix}.convs1.\{convolution}")
      append_audio_weight_norm(names, "\{prefix}.convs2.\{convolution}")
    }
  }
  for stage in 0..<7 {
    append_audio_weight_norm(names, "decoder.ups.\{stage}.0")
  }
  append_audio_weight_norm(names, "encoder.block.0")
  for stage in 1..<6 {
    let prefix = "encoder.block.\{stage}.block"
    for residual in 0..<3 {
      names.push("\{prefix}.\{residual}.block.0.alpha")
      append_audio_weight_norm(names, "\{prefix}.\{residual}.block.1")
      names.push("\{prefix}.\{residual}.block.2.alpha")
      append_audio_weight_norm(names, "\{prefix}.\{residual}.block.3")
    }
    names.push("\{prefix}.3.alpha")
    append_audio_weight_norm(names, "\{prefix}.4")
  }
  names.push("encoder.block.6.alpha")
  append_audio_weight_norm(names, "encoder.block.7")
  append_weight_bias(names, "logs_proj")
  append_weight_bias(names, "mean_proj")
  for
    name in [
      "pre_block.attn.proj.bias", "pre_block.attn.proj.weight", "pre_block.attn.q_bias",
      "pre_block.attn.qkv.weight", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias",
      "pre_block.mlp.norm.bias", "pre_block.mlp.norm.weight", "pre_block.mlp.w0.bias",
      "pre_block.mlp.w0.weight", "pre_block.mlp.w1.bias", "pre_block.mlp.w1.weight",
      "pre_block.mlp.w2.bias", "pre_block.mlp.w2.weight", "pre_block.norm1.bias",
      "pre_block.norm1.weight", "pre_block.norm2.bias", "pre_block.norm2.weight",
      "pre_block.norm3.bias", "pre_block.norm3.weight", "pre_block.proj.bias", "pre_block.proj.weight",
    ] {
    names.push(name)
  }
  names
}

///|
fn audio_vae_role(name : String) -> MiniMaxH3TensorRole? {
  guard audio_vae_shape(name) is Some(_) else { return None }
  if name.has_prefix("encoder.") {
    Some(AudioEncoder)
  } else if name.has_prefix("mean_proj.") || name.has_prefix("logs_proj.") {
    Some(AudioLatentProjection)
  } else {
    Some(AudioDecoder)
  }
}
