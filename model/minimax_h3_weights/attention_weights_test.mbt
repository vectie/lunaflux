///|
test "MiniMax H3 authenticates all 50 plus 2 bias-free attention weight sets" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  for index in 0..<52 {
    let prefix = if index < 50 {
      "transformer_blocks.\{index}"
    } else {
      "token_refiner.blocks.\{index - 50}"
    }
    for
      item in [
        (prefix + ".attn.to_q.weight", [7168L, 5376L]),
        (prefix + ".attn.to_k.weight", [7168L, 5376L]),
        (prefix + ".attn.to_v.weight", [7168L, 5376L]),
        (prefix + ".attn.to_out.0.weight", [5376L, 7168L]),
        (prefix + ".attn.norm_q.weight", [128L]),
        (prefix + ".attn.norm_k.weight", [128L]),
      ] {
      let (name, shape) = item
      let binding = @minimax_h3_weights.bind(
        metadata,
        plan,
        Ref2vaDenoiser,
        one_tensor(name, shape~),
      )
      let bound = binding.tensor(0).unwrap()
      assert_eq(bound.role(), DenoiserAttention)
      assert_eq(bound.tensor().shape(), ReadOnlyArray::from_array(shape))
      assert_eq(bound.tensor().dtype(), BFloat16)
    }
  }
}

///|
test "MiniMax H3 rejects attention shape dtype bias and layer substitutions" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  for
    hostile in [
      one_tensor("transformer_blocks.0.attn.to_q.weight", shape=[7167L, 5376L]),
      one_tensor(
        "transformer_blocks.0.attn.norm_q.weight",
        shape=[128L],
        dtype="F32",
      ),
    ] {
    try
      @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, hostile)
    catch {
      InvalidSemanticRank(index=0) | InvalidSemanticDType(index=0) => ()
      _ => fail("unexpected attention tensor rejection")
    } noraise {
      _ => fail("hostile attention tensor admitted")
    }
  }
  for
    name in [
      "transformer_blocks.0.attn.to_q.bias", "token_refiner.blocks.2.attn.to_q.weight",
    ] {
    let tensor = try! one_tensor(name, shape=[7168L, 5376L])
    try @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, tensor) catch {
      UnknownSemanticTensor(index=0) => ()
      _ => fail("unexpected attention vocabulary rejection")
    } noraise {
      _ => fail("bias or third refiner attention admitted")
    }
  }
}
