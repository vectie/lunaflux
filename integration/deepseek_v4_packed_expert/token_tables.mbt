///|
/// Resolve only the hash-routing tables used by this contiguous base-layer
/// interval. The source retains its exact I64 shape; startup upload owns checked
/// I32 storage rather than retaining a full-model host arena.
pub fn bind_token_hash_tables(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  first_layer~ : Int,
  end_layer~ : Int,
) -> Array[@materialize.CheckpointI64Table] raise {
  guard first_layer >= 0 &&
    end_layer > first_layer &&
    end_layer <= spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  [
    for layer in first_layer..<end_layer.min(spec.hash_layer_count()) => {
      checkpoint.bind_i64_table(
        payload_name="layers.\{layer}.ffn.gate.tid2eid",
        rows=spec.vocabulary_size(),
        columns=spec.experts_per_token(),
      )
    }
  ]
}
