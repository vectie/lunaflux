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
  selected_alias? : Bool = false,
) -> (Bytes, Bytes) raise Qwen3Bf16CandidateExportCliError {
  let recipe = qwen_compiler_attention_recipe(
    operation,
    "paged-attention-decode-functional-tile",
    set,
    compiled,
    frontier_sha256=frontier,
    variant_stable_id=compiled.functional().plan().candidate().stable_id(),
  )
  let ordinary_law = qwen_decode_physical_numeric_law(compiled.lowering())
  guard ordinary_law is Some(numeric_law) else {
    return (compiled.source().bytes(), recipe)
  }
  let shape = compiled.functional().plan().problem().shape()
  let (_, _, _, page) = shape.dimensions()
  let context = set.profile().max_page_table_entries() * page
  let partition_candidate = compiled.functional().plan().candidate().stable_id()
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
  let ordinary_source = if selected_alias {
    qwen_decode_selected_source(compiled)
  } else {
    compiled.source()
  }
  let ordinary = @utf8.decode(ordinary_source.bytes()) catch {
    _ => raise Publication
  }
  guard qwen_decode_physical_numeric_law(partitioned.lowering())
    is Some(partition_law) &&
    partition_law == numeric_law else {
    raise QwenDecodeSplitPreparation
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
  let (partial_symbol, merge_symbol) = partitioned.source().symbols()
  let text = (@utf8.decode(recipe) catch { _ => raise Publication })
    .replace(
      old="function_symbol=" + compiled.source().symbol().to_string(),
      new="function_symbol=" + ordinary_source.symbol().to_string(),
    )
    .replace(
      old="source_sha256=" + compiled.source().digest().as_hex(),
      new="source_sha256=" + digest,
    ) +
    "partition_count=8\npartition_candidate=\{partition_candidate}\npartition_numeric_law=\{partition_law}\n" +
    "partition_function_symbol=\{partial_symbol}\npartition_merge_function_symbol=\{merge_symbol}\n" +
    "partition_source_sha256=\{partitioned.source().digest().as_hex()}\npartition_lowering_sha256=\{partitioned.lowering().digest().as_hex()}\n" +
    "partition_key_value_tile_rows=\{partitioned.lowering().tile().1}\npartition_pipeline_stages=\{partitioned.lowering().pipeline().0}\n" +
    "partition_max_registers=\{partitioned.lowering().compiler_register_ceiling(set.compiler_policy().max_registers())}\n" +
    (if selected_alias {
      "frontier_function_symbol=\{compiled.source().symbol()}\nfrontier_source_sha256=\{compiled.source().digest().as_hex()}\nselected_serving_alias=true\n"
    } else {
      ""
    }) +
    "partition_grid=\{set.profile().max_query_rows()},\{partitioned.lowering().launch().1},8\npartition_merge_grid=\{set.profile().max_query_rows()},\{shape.dimensions().0},1\npartition_block=\{partitioned.lowering().launch().2},1,1\npartition_shared_memory_bytes=\{partitioned.lowering().launch().3}\npartition_workspace_bytes=\{partitioned.source().workspace().total_bytes()}\n"
  (source, @utf8.encode(text))
}

///|
/// A selected serving alias is a fresh immutable terminal realization of the
/// exact same plan. Candidate-qualified frontier symbols and their digests
/// remain intact for measurement; the production ABI uses a stable law name.
fn qwen_decode_selected_source(
  compiled : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_source.CudaAttentionTileSource raise Qwen3Bf16CandidateExportCliError {
  @luna_cuda_attention_tile_source.emit_cuda_attention_tile_source(
    compiled.lowering(),
    compiled.source().abi(),
    "lunaflux_attention_decode_tile_compiler_v1",
    1048576,
  ) catch {
    _ => raise QwenDecodeSplitPreparation
  }
}

///|
/// Numeric identity comes from the final executable physical program, not a
/// growing list of historical candidate numbers maintained in the exporter.
fn qwen_decode_physical_numeric_law(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> String? {
  let program = lowering.physical_program() catch { _ => return None }
  match program {
    BlockwiseKeyProgram(fold, _) => Some(fold.numeric_law())
    GroupedMatrixKeyProgram(fold, _) => Some(fold.numeric_law())
    _ => None
  }
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
