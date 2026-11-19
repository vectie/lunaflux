///|
// openbmb/MiniCPM5-2B revision 42433e90dad95123ab87729fd7011e7110629556.
// Semantic admission only; this fixture contains no model weights or GPU claim.
test "MiniCPM5 2B uses the existing dense Llama configuration" {
  let document =
    #|{"_name_or_path":"openbmb/MiniCPM5-2B","architectures":["LlamaForCausalLM"],"bos_token_id":0,"eos_token_id":[1,130073],"pad_token_id":1,"hidden_act":"silu","hidden_size":2048,"initializer_range":0.02,"intermediate_size":6144,"max_position_embeddings":131072,"model_type":"llama","num_attention_heads":16,"num_hidden_layers":42,"num_key_value_heads":2,"head_dim":128,"rms_norm_eps":1e-6,"rope_theta":5000000,"rope_scaling":null,"tie_word_embeddings":false,"torch_dtype":"bfloat16","transformers_version":"5.6.2","use_cache":true,"vocab_size":130560}
  let spec = parse_selected(document).spec()
  assert_eq(spec.hidden_size(), 2048)
  assert_eq(spec.layer_count(), 42)
  assert_eq(spec.kv_head_count(), 2)
  assert_eq(spec.head_dimension(), 128)
  assert_eq(spec.vocabulary_size(), 130560)
  for malformed in ["[]", "[-1]", "[1.5]", "[true]", "[2147483648]"] {
    let invalid = document.replace(old="[1,130073]", new=malformed)
    assert_true(
      try parse_selected(invalid) catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}
