///|
fn component_tag(
  component : @minimax_h3_weights.MiniMaxH3WeightComponent,
) -> Int {
  match component {
    Fl2vaDenoiser => 1
    Ref2vaDenoiser => 2
    TextConditioner => 3
    VideoVae => 4
    AudioVae => 5
  }
}

///|
fn role_tag(role : @minimax_h3_weights.MiniMaxH3TensorRole) -> Int {
  match role {
    DenoiserProjection => 1
    DenoiserTimeEmbedding => 2
    DenoiserAttention => 3
    DenoiserFeedForward => 4
    DenoiserNormalization => 5
    TextEmbedding => 6
    TextAttention => 7
    TextFeedForward => 8
    TextNormalization => 9
    VisionPatchEmbedding => 10
    VisionAttention => 11
    VisionFeedForward => 12
    VisionNormalization => 13
    VideoEncoder => 14
    VideoDecoder => 15
    VideoLatentProjection => 16
    AudioEncoder => 17
    AudioDecoder => 18
    AudioLatentProjection => 19
  }
}

///|
fn canonical_manifest_digest(
  component : @minimax_h3_weights.MiniMaxH3WeightComponent,
  identity : @spec.ModelIdentity,
  shards : ReadOnlyArray[MiniMaxH3ShardDeclarationInput],
  tensors : ReadOnlyArray[MiniMaxH3HostManifestTensor],
  arena_sizes : ReadOnlyArray[Int64],
  total_tensor_bytes : Int64,
) -> MiniMaxH3HostManifestDigest {
  let out = StringBuilder()
  out <+
    "lunaflux.minimax-h3-host-manifest.v1|component=\{component_tag(component)}|content=\{identity.content().as_hex()}|plan=\{identity.plan().as_hex()}"
  for shard_index, shard in shards {
    out <+
      "|shard=\{shard_index}:\{shard.file_digest.as_hex()}:\{shard.payload_bytes}"
  }
  for entry in tensors {
    let tensor = entry.tensor
    out <+
      "|tensor=\{entry.binding_index}:\{role_tag(entry.role)}:\{tensor.name()}:\{entry.shard_index}:\{tensor.data_start()}:\{tensor.data_end()}:\{entry.arena_index}:\{entry.arena_offset}"
    for dimension in tensor.shape() {
      out <+ ":\{dimension}"
    }
  }
  for arena_index, size in arena_sizes {
    out <+ "|arena=\{arena_index}:\{size}"
  }
  out <+ "|total=\{total_tensor_bytes}"
  {
    hex: @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(out.to_string())),
    ),
  }
}
