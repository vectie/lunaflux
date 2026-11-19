///|
/// Bind packed checkpoint tensors as expert-major gate/up/down slices for
/// integration/packed_expert_mlp.upload_bank. This is startup metadata only;
/// transfer_weights_to streams directly into the bank with bounded scratch.
/// Full source shape is retained and verified before flattening its leading axes.
pub fn bind_expert_weights(
  checkpoint : @checkpoint.PackedCheckpoint,
  plan : @precision.ExpertMlpPrecision,
  layer~ : Int32,
) -> Array[@checkpoint.PackedCheckpointWeight] raise {
  let (_, hidden, intermediate, experts, _) = plan.geometry()
  let gate_rows = experts.to_int64() * intermediate.to_int64() * 2L
  let down_rows = experts.to_int64() * hidden.to_int64()
  guard gate_rows <= 2147483647L &&
    down_rows <= 2147483647L &&
    plan.weights().all(w => w.representation().format() is Bf16) else {
    fail("unsupported BF16 expert storage")
  }
  let prefix = "model.language_model.layers.\{layer}.mlp.experts."
  let gate_representation = @format.Representation::new(
    format=Bf16,
    rows=gate_rows.to_int(),
    columns=hidden,
  )
  let down_representation = @format.Representation::new(
    format=Bf16,
    rows=down_rows.to_int(),
    columns=intermediate,
  )
  let gate_up = checkpoint.bind_dense_tensor(
    representation=gate_representation,
    payload_name=prefix + "gate_up_proj",
    source_shape=[
      experts.to_int64(),
      intermediate.to_int64() * 2L,
      hidden.to_int64(),
    ],
  )
  let down = checkpoint.bind_dense_tensor(
    representation=down_representation,
    payload_name=prefix + "down_proj",
    source_shape=[
      experts.to_int64(),
      hidden.to_int64(),
      intermediate.to_int64(),
    ],
  )
  let bound = []
  for expert in plan.local_experts() {
    for half in 0..<2 {
      bound.push(
        gate_up.slice(
          @precision.PackedMatrixSlice::new(
            gate_representation,
            row_start=(expert * 2 + half) * intermediate,
            rows=intermediate,
            column_start=0,
            columns=hidden,
          ),
        ),
      )
    }
    bound.push(
      down.slice(
        @precision.PackedMatrixSlice::new(
          down_representation,
          row_start=expert * hidden,
          rows=hidden,
          column_start=0,
          columns=intermediate,
        ),
      ),
    )
  }
  bound
}
