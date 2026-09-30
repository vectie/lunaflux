///|
/// One module contains the ordinary blockwise entry and its explicit
/// partition/merge alternative. The source digest binds all three symbols;
/// the runtime resolves them once and captures separate execution graphs.
fn qwen_decode_module_pair(
  operation : Int,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  layout : @device_layout.DeviceKvLayout,
  compiled : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
  frontier : String,
) -> (Bytes, Bytes) raise Qwen3Bf16CandidateExportCliError {
  let recipe = qwen_compiler_attention_recipe(
    operation,
    "paged-attention-decode-functional-tile",
    set,
    compiled,
    frontier_sha256=frontier,
    variant_stable_id=compiled.functional().plan().candidate().stable_id(),
  )
  let compute = compiled.functional().plan().candidate().compute()
  if compute != AttentionGroupedQueryBlockwiseDot &&
    compute != AttentionGroupedQueryDualScoreBlockwiseDot &&
    compute != AttentionGroupedQueryFmaBlockwiseDot {
    return (compiled.source().bytes(), recipe)
  }
  let shape = compiled.functional().plan().problem().shape()
  let (_, _, _, page) = shape.dimensions()
  let context = set.profile().max_page_table_entries() * page
  let partition_candidate = if compute == AttentionGroupedQueryFmaBlockwiseDot {
    456
  } else if compute == AttentionGroupedQueryDualScoreBlockwiseDot {
    454
  } else {
    452
  }
  let partitioned = @luna_cuda_paged_attention_aot.compile_blockwise_decode_split(
    shape,
    set.profile(),
    set.target(),
    layout.page_stride_bytes(),
    layout.total_page_count().to_int64(),
    context,
    8,
    268435456L,
    candidate_id=partition_candidate,
  ) catch {
    _ => raise QwenDecodeSplitPreparation
  }
  let ordinary = @utf8.decode(compiled.source().bytes()) catch {
    _ => raise Publication
  }
  let partition_law = match
    (partitioned.lowering().physical_program() catch {
      _ => raise QwenDecodeSplitPreparation
    }) {
    BlockwiseKeyProgram(fold, _) => fold.numeric_law()
    _ => raise QwenDecodeSplitPreparation
  }
  // Keep helper linkage and preprocessor state separate; extern-C kernels
  // retain their stable ABI while sharing one digest-pinned module.
  let reset = qwen_decode_module_macro_reset(ordinary)
  let source = @utf8.encode(
    ordinary +
    "\n" +
    reset +
    "\n#include <cuda_bf16.h>\n#include <math_constants.h>\nnamespace lf_blockwise_partition_v3 {\n" +
    (@utf8.decode(partitioned.source().bytes()) catch {
      _ => raise Publication
    }) +
    "\n}\n",
  )
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(source))
  let text = (@utf8.decode(recipe) catch { _ => raise Publication }).replace(
      old="source_sha256=" + compiled.source().digest().as_hex(),
      new="source_sha256=" + digest,
    ) +
    "partition_count=8\npartition_candidate=\{partition_candidate}\npartition_numeric_law=\{partition_law}\n" +
    "partition_grid=\{set.profile().max_query_rows()},\{partitioned.lowering().launch().1},8\npartition_merge_grid=\{set.profile().max_query_rows()},\{shape.dimensions().0},1\npartition_block=\{partitioned.lowering().launch().2},1,1\npartition_shared_memory_bytes=\{partitioned.lowering().launch().3}\npartition_workspace_bytes=\{partitioned.source().workspace().total_bytes()}\n"
  (source, @utf8.encode(text))
}

///|
/// Macro identities exclude function parameters. Resetting `NAME(item)`
/// instead of `NAME` leaves invalid extra tokens after a preprocessor undef.
fn qwen_decode_module_macro_reset(source : String) -> String {
  source
  .split("\n")
  .filter(line => line.has_prefix("#define LF_"))
  .map(line => {
    "#undef " +
    line.split(" ").to_array()[1].split("(").to_array()[0].to_owned()
  })
  .to_array()
  .join("\n")
}
