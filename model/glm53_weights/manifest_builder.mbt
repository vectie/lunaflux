///|
fn add_full_layer(
  tensors : Array[Glm53TensorDescriptor],
  layer : Int,
  spec : @glm53_spec.Glm53FullSpec,
) -> Unit {
  let prefix = "model.layers.\{layer}"
  add_tensor(tensors, "\{prefix}.input_layernorm.weight", BFloat16, [6144L])
  add_tensor(tensors, "\{prefix}.post_attention_layernorm.weight", BFloat16, [
    6144L,
  ])
  add_dsa(
    tensors,
    "\{prefix}.self_attn",
    6144L,
    64L,
    2048L,
    512L,
    192L,
    64L,
    256L,
    try! spec.computes_index_topk(layer),
    false,
  )
  if layer < 3 {
    add_dense_mlp(tensors, "\{prefix}.mlp", 6144L, 12288L)
  } else {
    add_moe(tensors, "\{prefix}.mlp", 6144L, 2048L, 256)
  }
}

///|
fn add_flash_layer(tensors : Array[Glm53TensorDescriptor], layer : Int) -> Unit {
  let prefix = "model.language_model.layers.\{layer}"
  add_tensor(tensors, "\{prefix}.input_layernorm.weight", BFloat16, [4096L])
  add_tensor(tensors, "\{prefix}.post_attention_layernorm.weight", BFloat16, [
    4096L,
  ])
  add_flash_hyper(tensors, prefix)
  if layer % 4 == 3 {
    add_dsa(
      tensors,
      "\{prefix}.self_attn",
      4096L,
      64L,
      1536L,
      512L,
      256L,
      0L,
      256L,
      true,
      true,
    )
  } else {
    add_flash_kda(tensors, "\{prefix}.self_attn")
  }
  if layer < 3 {
    add_dense_mlp(tensors, "\{prefix}.mlp", 4096L, 12288L)
  } else {
    add_moe(tensors, "\{prefix}.mlp", 4096L, 2048L, 288)
  }
}

///|
fn add_full_mtp(tensors : Array[Glm53TensorDescriptor]) -> Unit {
  let prefix = "model.layers.78"
  add_tensor(tensors, "\{prefix}.eh_proj.weight", BFloat16, [6144L, 12288L])
  add_tensor(tensors, "\{prefix}.enorm.weight", BFloat16, [6144L])
  add_tensor(tensors, "\{prefix}.hnorm.weight", BFloat16, [6144L])
  add_tensor(tensors, "\{prefix}.input_layernorm.weight", BFloat16, [6144L])
  add_tensor(tensors, "\{prefix}.post_attention_layernorm.weight", BFloat16, [
    6144L,
  ])
  add_dsa(
    tensors,
    "\{prefix}.self_attn",
    6144L,
    64L,
    2048L,
    512L,
    192L,
    64L,
    256L,
    true,
    false,
  )
  add_moe(tensors, "\{prefix}.mlp", 6144L, 2048L, 256)
  add_tensor(tensors, "\{prefix}.shared_head.norm.weight", BFloat16, [6144L])
}

///|
fn add_flash_mtp(tensors : Array[Glm53TensorDescriptor]) -> Unit {
  let prefix = "model.language_model.layers.45"
  add_tensor(tensors, "\{prefix}.eh_proj.weight", BFloat16, [4096L, 8192L])
  add_tensor(tensors, "\{prefix}.enorm.weight", BFloat16, [4096L])
  add_tensor(tensors, "\{prefix}.hnorm.weight", BFloat16, [4096L])
  add_tensor(tensors, "\{prefix}.input_layernorm.weight", BFloat16, [4096L])
  add_tensor(tensors, "\{prefix}.post_attention_layernorm.weight", BFloat16, [
    4096L,
  ])
  add_dsa(
    tensors,
    "\{prefix}.self_attn",
    4096L,
    64L,
    1536L,
    512L,
    256L,
    0L,
    256L,
    true,
    true,
  )
  add_moe(tensors, "\{prefix}.mlp", 4096L, 2048L, 288)
  add_tensor(tensors, "\{prefix}.shared_head.norm.weight", BFloat16, [4096L])
}

///|
/// Builds the exact logical full-model namespace for a BF16-converted
/// checkpoint. The official FP8 artifact remains fail-closed until its
/// per-tensor exclusion/scale layout has a materializer contract.
pub fn build_full_tensor_manifest(
  metadata : @glm53_spec.Glm53FullMetadata,
  encoding : Glm53WeightEncoding,
) -> Glm53TensorManifest raise Glm53WeightsError {
  if encoding == Fp8E4M3Block128Dynamic {
    raise UnsupportedPhysicalEncoding(encoding)
  }
  let tensors = []
  add_tensor(tensors, "model.embed_tokens.weight", BFloat16, [154880L, 6144L])
  for layer in 0..<78 {
    add_full_layer(tensors, layer, metadata.spec())
  }
  add_full_mtp(tensors)
  add_tensor(tensors, "model.norm.weight", BFloat16, [6144L])
  add_tensor(tensors, "lm_head.weight", BFloat16, [154880L, 6144L])
  finish_manifest(
    FullText,
    encoding,
    @glm53.build_full_execution_plan(metadata).identity(),
    tensors,
    [],
  )
}

///|
/// Builds the exact official GLM-5.3-Flash-BF16 text, MTP, KDA/DSA/mHC,
/// vision, projector, and embedding-backed control-token namespace.
pub fn build_flash_tensor_manifest(
  metadata : @glm53_spec.Glm53FlashMetadata,
  encoding : Glm53WeightEncoding,
) -> Glm53TensorManifest raise Glm53WeightsError {
  if encoding != BFloat16WithFloat32Routing {
    raise UnsupportedPhysicalEncoding(encoding)
  }
  let tensors = []
  add_tensor(tensors, "model.language_model.embed_tokens.weight", BFloat16, [
    154880L, 4096L,
  ])
  for layer in 0..<45 {
    add_flash_layer(tensors, layer)
  }
  add_flash_mtp(tensors)
  add_tensor(tensors, "model.language_model.norm.weight", BFloat16, [4096L])
  add_tensor(tensors, "lm_head.weight", BFloat16, [154880L, 4096L])
  add_flash_vision(tensors)
  let spec = metadata.spec()
  finish_manifest(
    FlashMultimodal,
    encoding,
    @glm53.build_flash_execution_plan(metadata).identity(),
    tensors,
    [
      spec.image_start_token_id(),
      spec.image_end_token_id(),
      spec.video_start_token_id(),
      spec.video_end_token_id(),
      spec.image_token_id(),
      spec.video_token_id(),
    ],
  )
}
