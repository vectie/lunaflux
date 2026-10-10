///|
fn append_video_vae_pair(names : Array[String], prefix : String) -> Unit {
  names.push("\{prefix}.weight")
  names.push("\{prefix}.bias")
}

///|
fn expected_video_vae_names() -> Array[String] {
  let names : Array[String] = []
  append_video_vae_pair(names, "decoder.norm_out")
  append_video_vae_pair(names, "decoder.proj_in")
  append_video_vae_pair(names, "decoder.proj_out")
  names.push("decoder.register_tokens")
  for layer in 0..<36 {
    let prefix = "decoder.transformer_blocks.\{layer}"
    for projection in ["to_k", "to_q", "to_v"] {
      append_video_vae_pair(names, "\{prefix}.attn.\{projection}")
    }
    append_video_vae_pair(names, "\{prefix}.attn.to_out.0")
    append_video_vae_pair(names, "\{prefix}.ff.net.0.proj")
    append_video_vae_pair(names, "\{prefix}.ff.net.2")
    names.push("\{prefix}.norm1.weight")
    names.push("\{prefix}.norm2.weight")
    names.push("\{prefix}.scale1")
    names.push("\{prefix}.scale2")
  }
  append_video_vae_pair(names, "encoder.conv_in")
  for block in 0..<6 {
    let prefix = "encoder.down_blocks.\{block}"
    for resnet in 0..<2 {
      let resnet_prefix = "\{prefix}.resnets.\{resnet}"
      append_video_vae_pair(names, "\{resnet_prefix}.conv1")
      append_video_vae_pair(names, "\{resnet_prefix}.conv2")
      append_video_vae_pair(names, "\{resnet_prefix}.norm1")
      append_video_vae_pair(names, "\{resnet_prefix}.norm2")
      if resnet == 0 && (block == 1 || block == 3 || block == 5) {
        append_video_vae_pair(names, "\{resnet_prefix}.conv_shortcut")
      }
    }
    if block < 4 {
      append_video_vae_pair(names, "\{prefix}.downsamplers.0.conv")
    }
  }
  append_video_vae_pair(names, "encoder.conv_out")
  append_video_vae_pair(names, "encoder.norm_out")
  append_video_vae_pair(names, "post_quant_conv")
  append_video_vae_pair(names, "quant_conv")
  names
}

///|
fn video_vae_role(name : String) -> MiniMaxH3TensorRole? {
  guard video_vae_shape(name) is Some(_) ||
    original_video_shape(name) is Some(_) else {
    return None
  }
  if name.has_prefix("decoder.") {
    Some(VideoDecoder)
  } else if name.has_prefix("encoder.") {
    Some(VideoEncoder)
  } else {
    Some(VideoLatentProjection)
  }
}
