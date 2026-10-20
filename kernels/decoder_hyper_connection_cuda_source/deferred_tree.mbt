///|
/// Preserve the declared descending tree while moving its suffix into one
/// subgroup. Other subgroups do not read or overwrite the shared tree until
/// the publishing subgroup has completed.
fn deferred_tree_source(
  workers : Int,
  groups : Int,
  subgroup_width : Int,
) -> String raise {
  let (shared, local) = @physical.ReductionTreePartition::new(
    workers, subgroup_width,
  ).levels()
  let text = StringBuilder()
  for stride in shared {
    text <+
      "if(worker<\{stride}) { \n#pragma unroll\nfor(int c=0;c<\{groups};++c) reduction[c][worker]=__fadd_rn(reduction[c][worker],reduction[c][worker+\{stride}]); }\n__syncthreads();\n"
  }
  if !local.is_empty() {
    text <+
      "if(worker<\{subgroup_width}) {\n#pragma unroll\nfor(int c=0;c<\{groups};++c) {\nfloat folded=reduction[c][worker];\n"
    for stride in local {
      text <+
        "const float peer\{stride}=__shfl_down_sync(0xffffffffU,folded,\{stride}); if(worker<\{stride}) folded=__fadd_rn(folded,peer\{stride});\n"
    }
    text <+ "if(worker==0) reduction[c][0]=folded;\n}\n}\n__syncthreads();\n"
  }
  text.to_string()
}
