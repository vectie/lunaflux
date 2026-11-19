///|
pub(all) struct MoeIndices {
  router : Int32
  routed_bank : Int32
  shared_gate : Int32
  shared_up : Int32
  shared_down : Int32
  shared_gate_projection : Int32
}

///|
pub(all) struct RecurrentIndices {
  qkv : Int32
  gate : Int32
  decay : Int32
  beta : Int32
  output : Int32
  convolution : Int32
  norm : Int32
  a_log : Int32
  dt_bias : Int32
}

///|
pub(all) struct AttentionIndices {
  query : Int32
  key : Int32
  value : Int32
  output : Int32
  query_norm : Int32
  key_norm : Int32
}

///|
pub(all) enum BranchIndices {
  Recurrent(RecurrentIndices)
  FullAttention(AttentionIndices)
}

///|
pub(all) struct LayerIndices {
  input_norm : Int32
  post_attention_norm : Int32
  branch : BranchIndices
  moe : MoeIndices
}

///|
pub(all) struct TextIndices {
  embedding : Int32
  final_norm : Int32
  language_head : Int32
}
