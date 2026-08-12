///|
/// Validates the complete selected dense-Llama artifact contract before any
/// host or device weight allocation.
///
/// `content_digest` must have been verified by the bounded artifact reader.
/// Plan identity is derived internally from the admitted semantic model spec.
pub fn admit(
  config : BytesView,
  safetensors_file : BytesView,
  content_digest : @spec.ContentDigest,
  config_max_bytes~ : Int,
  config_max_depth~ : Int,
  header_limits~ : @safetensors.HeaderLimits,
) -> LlamaAdmission raise LlamaAdmissionError {
  let metadata = @config_reader.parse_bytes(
    config,
    content_digest,
    max_bytes=config_max_bytes,
    max_depth=config_max_depth,
  ) catch {
    error => raise Configuration(error)
  }
  let safetensors = @safetensors_reader.read_metadata(
    safetensors_file,
    limits=header_limits,
  ) catch {
    error => raise Safetensors(error)
  }
  complete(metadata, safetensors)
}

///|
/// Completes selected dense-Llama admission from already validated model and
/// safetensors metadata without reparsing their source bytes.
pub fn complete(
  metadata : @spec.LlamaModelMetadata,
  safetensors : @safetensors.SafetensorsMetadata,
) -> LlamaAdmission raise LlamaAdmissionError {
  let plan = @llama.build(metadata) catch { error => raise Plan(error) }
  let bindings = @llama_weights.bind(metadata, safetensors) catch {
    error => raise Weights(error)
  }
  { metadata, safetensors, plan, bindings }
}
