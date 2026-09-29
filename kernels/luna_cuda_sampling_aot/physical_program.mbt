///|
fn sampling_selection_program(
  values : Int,
  segmented~ : Bool,
) -> @elementwise_physical_ir.SelectionProgram raise SamplingBoundaryError {
  guard @elementwise_physical_ir.SelectionProgram::new(
      values~,
      owners=256,
      subgroup_lanes=32,
      reduction=if segmented { SubgroupTree } else { OwnerTree },
    )
    is Some(program) else {
    raise Invalid(VocabularySize)
  }
  program
}

///|
fn greedy_owner_reduction_source(
  program : @elementwise_physical_ir.SelectionProgram,
) -> String {
  let source = StringBuilder()
  for effect in program.effects() {
    source.write_string(
      match effect {
        PublishOwnerSummaries =>
          "  values[lane] = best; tokens[lane] = best_token; invalids[lane] = invalid;\n"
        SynchronizeSelection => "  __syncthreads();\n"
        ReduceOwnerSummaries(stride) =>
          "  for (int stride = \{stride}; stride > 0; stride >>= 1) {\n" +
          "    if (lane < stride) {\n      const float other = values[lane + stride];\n      const int other_token = tokens[lane + stride];\n" +
          "      if (other > values[lane] || (other == values[lane] && other_token < tokens[lane])) { values[lane] = other; tokens[lane] = other_token; }\n" +
          "      if (invalids[lane + stride] < invalids[lane]) invalids[lane] = invalids[lane + stride];\n    }\n    __syncthreads();\n  }\n"
        StoreSelection =>
          "  if (lane == 0) { const int bad = invalids[0]; results[row * 2] = bad == INT_MAX ? tokens[0] : -1; results[row * 2 + 1] = bad == INT_MAX ? -1 : bad; }\n"
        ScanOwnedValues => ""
        ReduceSubgroupSummaries(_)
        | PublishSubgroupSummaries
        | ReducePublishedSubgroups(_) =>
          abort("greedy owner lowering requires owner tree")
      },
    )
  }
  source.to_string()
}

///|
fn greedy_subgroup_reduction_source(
  program : @elementwise_physical_ir.SelectionProgram,
) -> String {
  let (_, owners, lanes) = program.geometry()
  let source = StringBuilder()
  source.write_string(
    "  __shared__ LunaGreedySummaryV2 warp_summaries[\{owners / lanes}];\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n",
  )
  for effect in program.effects() {
    source.write_string(
      match effect {
        ReduceSubgroupSummaries(stride) =>
          "  for (int delta=\{stride}; delta>0; delta>>=1) {\n" +
          "    float other=__shfl_down_sync(0xffffffffU,value,delta);\n" +
          "    int other_token=__shfl_down_sync(0xffffffffU,token,delta);\n" +
          "    int other_invalid=__shfl_down_sync(0xffffffffU,invalid,delta);\n" +
          "    if (lane+delta<\{lanes}) luna_greedy_combine_v2(value,token,invalid,other,other_token,other_invalid);\n  }\n"
        PublishSubgroupSummaries =>
          "  if (lane==0) warp_summaries[warp]={value,token,invalid,0};\n"
        SynchronizeSelection => "  __syncthreads();\n"
        ReducePublishedSubgroups(stride) =>
          "  if (warp==0) {\n" +
          "    value=lane<\{owners / lanes}?warp_summaries[lane].value:-INFINITY;\n" +
          "    token=lane<\{owners / lanes}?warp_summaries[lane].token:INT_MAX;\n" +
          "    invalid=lane<\{owners / lanes}?warp_summaries[lane].invalid:INT_MAX;\n" +
          "    for(int delta=\{stride};delta>0;delta>>=1) {\n" +
          "      float other=__shfl_down_sync(0xffffffffU,value,delta);\n" +
          "      int other_token=__shfl_down_sync(0xffffffffU,token,delta);\n" +
          "      int other_invalid=__shfl_down_sync(0xffffffffU,invalid,delta);\n" +
          "      if(lane+delta<\{lanes}) luna_greedy_combine_v2(value,token,invalid,other,other_token,other_invalid);\n    }\n  }\n"
        ScanOwnedValues | StoreSelection => ""
        PublishOwnerSummaries | ReduceOwnerSummaries(_) =>
          abort("segmented greedy requires subgroup tree")
      },
    )
  }
  source.to_string()
}
