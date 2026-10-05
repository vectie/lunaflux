///|
/// Final realization of the typed online-fold operations. The program owns
/// order and state dependencies; this backend owns fragment/lane instructions.
fn cuda_online_fold_phase(
  operations : ReadOnlyArray[@attention_physical_ir.FoldOperation],
  fragments : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
) -> String {
  let (score_rows, score_columns, _, score_quantum, _) = fragments
    .score()
    .program()
    .geometry()
  let (value_rows, value_columns, _, value_quantum, _) = fragments
    .value()
    .program()
    .geometry()
  // The retained binding owns scalarized product counts. These are not new
  // reduction schedules, and each reusable left fragment is still loaded once.
  guard score_rows == 1 && value_rows == 1 else {
    abort("invalid single-use fragment binding")
  }
  operations
  .map(operation => {
    match operation {
      ScoreProduct =>
        if fragments.query_lifetime().retained_fragments() > 0 ||
          fragments.query_lifetime().joint_right_consumers() {
          cuda_retained_query_score_product(fragments)
        } else {
          (
            $|        #pragma unroll
            $|        for (int reduction=0; reduction<LF_HEAD_DIMENSION; reduction+=\{score_quantum}) {
            $|          const auto a=lf_matrix_load(qbuf,qr,reduction,LF_HEAD_DIMENSION);
            $|          #pragma unroll
            $|          for (int key=0; key<\{score_columns}; ++key) {
            $|            lf_matrix_mma_key(score[key],a,current_k,key*16,reduction,LF_HEAD_DIMENSION);
            $|          }
            $|        }
          )
        }
      PrepareScores =>
        (
          #|        float next[2]={-CUDART_INF_F,-CUDART_INF_F};
          #|        const bool interior=__all_sync(0xffffffffU,key_end-kb>=LF_KEY_VALUE_TILE && min(p0,p1)-kb>=LF_KEY_VALUE_TILE-1);
          #|        if (interior) lf_query_score_fold<false>(score,next,scale,kb,key_end,p0,p1,lane);
          #|        else lf_query_score_fold<true>(score,next,scale,kb,key_end,p0,p1,lane);
          #|        lf_query_score_maximum(score,next);
        )
      MergeMaximum =>
        (
          #|        float alpha[2], sum[2]={};
          #|        #pragma unroll
          #|        for (int r=0;r<2;++r) {
          #|          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffU,next[r],1,4));
          #|          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffU,next[r],2,4));
          #|          next[r]=fmaxf(next[r],maximum[r]);
          #|          alpha[r]=lf_masked_exp(maximum[r],maximum[r]-next[r]);
          #|          maximum[r]=next[r];
          #|        }
        )
      RescaleOutput =>
        (
          #|        #pragma unroll
          #|        for (int column=0;column<LF_HEAD_DIMENSION/16;++column) {
          #|          #pragma unroll
          #|          for (int item=0;item<8;++item) out[column].x[item]*=alpha[item%4/2];
          #|        }
        )
      AccumulateProbabilityValue =>
        cuda_streaming_probability_value(
          fragments, value_columns, value_quantum,
        )
      UpdateDenominator =>
        (
          #|        #pragma unroll
          #|        for (int r=0;r<2;++r) {
          #|          sum[r]+=__shfl_xor_sync(0xffffffffU,sum[r],1,4);
          #|          sum[r]+=__shfl_xor_sync(0xffffffffU,sum[r],2,4);
          #|          denominator[r]=denominator[r]*alpha[r]+sum[r];
          #|        }
        )
    }
  })
  .join("\n")
}
