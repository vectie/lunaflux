///|
async test "actual packed expert shards map into one contiguous bank without transpose" {
  let (spec, shards) = source()
  let weights = @minicpm_v47_weights.bind(spec, shards[:])
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let plan = operators.routed_experts()
  assert_eq(plan.bank_bytes(), 1610612736L)
  assert_eq(plan.expert_stride(), 6291456L)
  let full_projections : Array[(String, @operators.FullProjection)] = [
    ("q_proj", QueryAndGate),
    ("k_proj", Key),
    ("v_proj", Value),
    ("o_proj", Output),
  ]
  for layer in [3, 39] {
    for (name, site) in full_projections {
      let (_, columns, inner) = operators.full_projection(site).geometry()
      let binding = weights
        .find_tensor(
          "model.language_model.layers.\{layer}.self_attn." + name + ".weight",
        )
        .unwrap()
      assert_eq(
        binding.shape,
        ReadOnlyArray::from_array([columns.to_int64(), inner.to_int64()]),
      )
    }
  }
  for layer in [0, 3, 39] {
    let prefix = "model.language_model.layers.\{layer}.mlp.experts."
    let gate_up = weights.find_tensor(prefix + "gate_up_proj").unwrap()
    let down = weights.find_tensor(prefix + "down_proj").unwrap()
    let copies = operators.expert_uploads(weights, layer~)
    assert_eq(copies.length(), 768)
    assert_eq(copies[0].file_offset, gate_up.file_offset)
    assert_eq(copies[1].file_offset, gate_up.file_offset + 2097152L)
    assert_eq(copies[2].file_offset, down.file_offset)
    assert_eq(copies[765].file_offset, gate_up.file_offset + 1069547520L)
    assert_eq(
      copies[767].file_offset + copies[767].byte_length,
      down.file_offset + down.byte_length,
    )
    let mut previous_end = 0L
    for index, copy in copies {
      assert_eq(copy.bank_offset, previous_end)
      assert_eq(copy.byte_length, 2097152L)
      let owner = if index % 3 < 2 { gate_up } else { down }
      assert_eq(copy.shard_index, owner.shard_index)
      assert_true(
        copy.file_offset >= owner.file_offset &&
        copy.file_offset + copy.byte_length <=
        owner.file_offset + owner.byte_length,
      )
      previous_end += copy.byte_length
    }
    assert_eq(previous_end, plan.bank_bytes())
  }
}
