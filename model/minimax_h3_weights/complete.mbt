///|
fn complete_vocabulary(component : MiniMaxH3WeightComponent) -> Array[String] {
  match component {
    Fl2vaDenoiser | Ref2vaDenoiser => expected_denoiser_names()
    TextConditioner => expected_text_names()
    VideoVae => expected_video_vae_names()
    AudioVae => expected_audio_vae_names()
  }
}

///|
/// Joins validated component shards into one exact schema-v1 component.
/// Successful construction proves every required source name is present
/// exactly once and every shard carries the same content/plan identity.
///
/// Every supported component completes only against its exact vocabulary.
pub fn join_complete_component(
  component : MiniMaxH3WeightComponent,
  shards : ArrayView[MiniMaxH3PartialShardBinding],
) -> MiniMaxH3CompleteComponentBinding raise MiniMaxH3WeightsError {
  if shards.is_empty() {
    raise IncompleteComponent(
      expected=component.required_tensor_count(),
      actual=0,
    )
  }
  let identity = shards[0].identity
  let schema_version = shards[0].schema_version
  let all : Array[MiniMaxH3SemanticTensorBinding] = []
  for shard_index, shard in shards {
    if shard.component != component {
      raise BindingComponentMismatch(index=shard_index)
    }
    if shard.identity != identity || shard.schema_version != schema_version {
      raise BindingIdentityMismatch(index=shard_index)
    }
    for tensor in shard.tensors {
      all.push(tensor)
    }
  }
  // Original and converted names may share normalization tensors, but their
  // complete vocabularies must never be mixed or silently treated as aliases.
  let original_video = component == VideoVae &&
    all.any(t => {
      original_video_shape(t.tensor.name()) is Some(_) &&
      video_vae_shape(t.tensor.name()) is None
    })
  let expected = if original_video {
    expected_original_video_names()
  } else {
    complete_vocabulary(component)
  }
  if all.length() != expected.length() {
    raise IncompleteComponent(expected=expected.length(), actual=all.length())
  }
  let canonical : Array[MiniMaxH3SemanticTensorBinding] = []
  for expected_name in expected {
    let mut found : MiniMaxH3SemanticTensorBinding? = None
    for tensor in all {
      if tensor.tensor.name() == expected_name {
        match found {
          Some(_) => raise DuplicateSemanticTensor(index=tensor.source_index)
          None => found = Some(tensor)
        }
      }
    }
    guard found is Some(binding) else {
      raise IncompleteComponent(expected=expected.length(), actual=all.length())
    }
    canonical.push(binding)
  }
  {
    component,
    identity,
    schema_version,
    tensors: ReadOnlyArray::from_array(canonical),
  }
}
