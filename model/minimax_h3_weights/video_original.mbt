///|
/// Original AutoencoderKLLegacy checkpoint layout. Semantic execution still
/// uses the converted decoder vocabulary; source names and QKV slices are
/// translated once during checkpoint packing, never inside a request.
pub struct MiniMaxH3OriginalVideoVaeSchema {
  priv version : Int
} derive(Debug, Eq)

///|
pub fn MiniMaxH3OriginalVideoVaeSchema::official_v1() -> Self {
  { version: 1, }
}

///|
fn original_video_name(canonical : String) -> String {
  canonical
  .replace(old="decoder.proj_in.", new="decoder.x_embedder.")
  .replace(old=".attn.to_out.0.", new=".attn.to_out.")
  .replace(old=".ff.net.0.proj.", new=".ff.w1.")
  .replace(old=".ff.net.2.", new=".ff.w2.")
  .replace(old="encoder.down_blocks.", new="encoder.down.")
  .replace(old=".resnets.", new=".block.")
  .replace(old=".conv_shortcut.", new=".nin_shortcut.")
  .replace(old=".downsamplers.0.conv.", new=".downsample.conv.")
}

///|
fn canonical_video_name(original : String) -> String {
  original
  .replace(old="decoder.x_embedder.", new="decoder.proj_in.")
  .replace(old=".attn.to_out.", new=".attn.to_out.0.")
  .replace(old=".ff.w1.", new=".ff.net.0.proj.")
  .replace(old=".ff.w2.", new=".ff.net.2.")
  .replace(old="encoder.down.", new="encoder.down_blocks.")
  .replace(old=".block.", new=".resnets.")
  .replace(old=".nin_shortcut.", new=".conv_shortcut.")
  .replace(old=".downsample.conv.", new=".downsamplers.0.conv.")
}

///|
fn original_video_shape(name : String) -> Array[Int64]? {
  if name == "decoder.mask_token" {
    return Some([1L, 1L, 2048L])
  }
  let parts = name.split(".").to_array()
  if parts.length() == 6 &&
    parts[0] == "decoder" &&
    parts[1] == "transformer_blocks" &&
    bounded_index(parts[2], 36) &&
    parts[3] == "attn" &&
    parts[4] == "to_qkv" {
    return match parts[5] {
      "weight" => Some([6144L, 2048L])
      "bias" => Some([6144L])
      _ => None
    }
  }
  let canonical = canonical_video_name(name)
  if original_video_name(canonical) != name ||
    canonical.contains(".attn.to_q.") ||
    canonical.contains(".attn.to_k.") ||
    canonical.contains(".attn.to_v.") {
    return None
  }
  video_vae_shape(canonical)
}

///|
fn expected_original_video_names() -> Array[String] {
  let names = ["decoder.mask_token"]
  for canonical in expected_video_vae_names() {
    if canonical.contains(".attn.to_k.") || canonical.contains(".attn.to_v.") {
      continue
    }
    names.push(
      original_video_name(canonical).replace(
        old=".attn.to_q.",
        new=".attn.to_qkv.",
      ),
    )
  }
  names
}

///|
pub fn MiniMaxH3OriginalVideoVaeSchema::tensor_count(_self : Self) -> Int {
  560
}

///|
pub fn MiniMaxH3OriginalVideoVaeSchema::tensor_name(
  self : Self,
  index : Int,
) -> String? {
  if index < 0 || index >= self.tensor_count() {
    None
  } else {
    Some(expected_original_video_names()[index])
  }
}

///|
pub fn MiniMaxH3OriginalVideoVaeSchema::tensor_shape(
  self : Self,
  index : Int,
) -> ReadOnlyArray[Int64]? {
  guard self.tensor_name(index) is Some(name) &&
    original_video_shape(name) is Some(shape) else {
    return None
  }
  Some(ReadOnlyArray::from_array(shape))
}

///|
pub fn MiniMaxH3OriginalVideoVaeSchema::matches_tensor(
  _self : Self,
  name : String,
  dtype : @safetensors.TensorDType,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  dtype == Float32 &&
  (match original_video_shape(name) {
    Some(expected) => video_vae_shape_is(shape, expected)
    None => false
  })
}

///|
/// A whole-tensor alias is available only when no headwise slicing is needed.
pub fn original_video_vae_source_name(canonical : String) -> String? {
  guard video_vae_shape(canonical) is Some(_) &&
    !canonical.contains(".attn.to_q.") &&
    !canonical.contains(".attn.to_k.") &&
    !canonical.contains(".attn.to_v.") else {
    return None
  }
  Some(original_video_name(canonical))
}

///|
/// Preserve canonical destination geometry while reading original
/// [head,Q/K/V,channel] rows. Upstream reshapes to [heads,3*head_dim] before
/// chunking, so whole-matrix Q/K/V slicing would silently load wrong weights.
pub fn original_video_vae_copies(
  copy : BlockWeightCopy,
) -> Array[BlockWeightCopy] raise BlockPackingError {
  guard video_vae_shape(copy.tensor_name) is Some(shape) else {
    raise InvalidBlock
  }
  let bytes = shape.fold(init=4L, (n, d) => n * d)
  guard copy.source_offset >= 0L &&
    copy.byte_count > 0L &&
    copy.byte_count <= bytes &&
    copy.source_offset <= bytes - copy.byte_count else {
    raise BudgetExceeded
  }
  let projection = if copy.tensor_name.contains(".attn.to_q.") {
    Some((copy.tensor_name.replace(old=".attn.to_q.", new=".attn.to_qkv."), 0L))
  } else if copy.tensor_name.contains(".attn.to_k.") {
    Some((copy.tensor_name.replace(old=".attn.to_k.", new=".attn.to_qkv."), 1L))
  } else if copy.tensor_name.contains(".attn.to_v.") {
    Some((copy.tensor_name.replace(old=".attn.to_v.", new=".attn.to_qkv."), 2L))
  } else {
    None
  }
  guard projection is Some((name, segment)) else {
    return [
      {
        tensor_name: original_video_name(copy.tensor_name),
        source_offset: copy.source_offset,
        destination_offset: copy.destination_offset,
        byte_count: copy.byte_count,
      },
    ]
  }
  let head_bytes = bytes / 32L
  let end = copy.source_offset + copy.byte_count
  let slices : Array[BlockWeightCopy] = []
  for head in 0..<32 {
    let head_start = head.to_int64() * head_bytes
    let start = Int64::max(head_start, copy.source_offset)
    let stop = Int64::min(head_start + head_bytes, end)
    if stop > start {
      slices.push({
        tensor_name: name,
        source_offset: (head.to_int64() * 3L + segment) * head_bytes +
        start -
        head_start,
        destination_offset: copy.destination_offset + start - copy.source_offset,
        byte_count: stop - start,
      })
    }
  }
  slices
}
