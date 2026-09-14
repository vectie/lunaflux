///|
async fn run_qwen3_candidate_export(
  raw_arguments : ArrayView[String],
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  let arguments = parse_qwen3_candidate_export_arguments(raw_arguments)
  let plan = load_authenticated_paged_plan(arguments)
  let layout = prepare_device_layout(plan, arguments)
  let fold_input = load_attention_offline_input(
    arguments.projection_folds,
    arguments.toolchain_sha256,
  )
  let fold_records : ReadOnlyArray[@luna_projection_tuning.ProjectionFoldRecord] = match
    fold_input {
    None => []
    Some((bytes, device, toolchain)) =>
      @luna_projection_tuning.parse_projection_fold_records(
        bytes,
        scope=device + "\t" + toolchain,
      ) catch {
        _ => raise InvalidArguments
      }
  }
  let set = prepare_candidate_set(
    plan,
    layout,
    arguments,
    fold_records=fold_records[:],
  )
  let residual = prepare_residual_rmsnorm_candidate(plan, set)
  let production = @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_production_candidate(
    residual,
    @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_promotion_bytes=65536,
    ),
  ) catch {
    _ => raise ResidualProductionPreparation
  }
  let tuning = load_attention_offline_input(
    arguments.attention_tuning,
    arguments.toolchain_sha256,
  )
  let resources = load_attention_offline_input(
    arguments.attention_resources,
    arguments.toolchain_sha256,
  )
  let qwen_fused = prepare_qwen_fused_production_set(
    plan,
    layout,
    set,
    tuning~,
    resources~,
    decode_resources=load_attention_offline_input(
      arguments.decode_resources,
      arguments.toolchain_sha256,
    ),
    decode_tuning=load_attention_offline_input(
      arguments.decode_tuning,
      arguments.toolchain_sha256,
    ),
    partitioned_resources=load_attention_offline_input(
      arguments.partitioned_resources,
      arguments.toolchain_sha256,
    ),
    query_metadata=arguments.query_metadata,
  )
  let evidence = @luna_bf16_candidate_export.export_new(
    set,
    arguments.absolute_new_output,
  ) catch {
    _ => raise Publication
  }
  let fused_root = arguments.absolute_new_output + "/reusable-fused-residual"
  write_new_candidate_pair(
    fused_root,
    production.source_bytes(),
    production.recipe_bytes(),
  )
  let partitioned_prefill_variants = qwen_fused.attention_prefill_partitioned.variants()
  for index, variant in partitioned_prefill_variants {
    let compilation = variant.compilation()
    let variant_root = qwen_partitioned_prefill_variant_root(
      arguments.absolute_new_output,
      partitioned_prefill_variants,
      index,
    )
    write_new_candidate_pair(
      variant_root,
      compilation.source().bytes(),
      qwen_compiler_partitioned_attention_recipe(
        qwen_fused.first_operation + 3,
        set,
        compilation,
        frontier_sha256=qwen_fused.attention_prefill_partitioned
          .digest()
          .as_hex(),
      ),
    )
  }
  let ingress_root = arguments.absolute_new_output + "/reusable-qwen-ingress"
  write_new_candidate_pair(
    ingress_root,
    qwen_fused.postprocess.source_bytes(),
    qwen_fused.postprocess.recipe_bytes(),
  )
  let full_ingress_root = arguments.absolute_new_output +
    "/reusable-qwen-full-ingress"
  write_new_candidate_pair(
    full_ingress_root,
    qwen_fused.full_ingress.source_bytes(),
    qwen_fused.full_ingress.recipe_bytes(),
  )
  let readonly_root = arguments.absolute_new_output +
    "/reusable-qwen-readonly-attention"
  write_new_candidate_pair(
    readonly_root,
    qwen_fused.attention_read.source_bytes(),
    qwen_fused.attention_read.recipe_bytes(),
  )
  let prefill_root = arguments.absolute_new_output +
    "/reusable-qwen-prefill-attention"
  let selected_prefill = qwen_fused.attention_prefill_frontier.selected()
  write_new_candidate_pair(
    prefill_root,
    selected_prefill.source().bytes(),
    qwen_compiler_attention_recipe(
      qwen_fused.first_operation + 3,
      "paged-attention-prefill-functional-tile",
      set,
      selected_prefill,
      frontier_sha256=qwen_fused.attention_prefill_frontier.digest().as_hex(),
      variant_stable_id=selected_prefill
        .functional()
        .plan()
        .candidate()
        .stable_id(),
    ),
  )
  for variant in qwen_fused.attention_prefill_frontier.variants() {
    let stable_id = variant.functional().plan().candidate().stable_id()
    let variant_root = arguments.absolute_new_output +
      "/reusable-qwen-prefill-attention-c" +
      stable_id.to_string()
    write_new_candidate_pair(
      variant_root,
      variant.source().bytes(),
      qwen_compiler_attention_recipe(
        qwen_fused.first_operation + 3,
        "paged-attention-prefill-functional-tile-frontier",
        set,
        variant,
        frontier_sha256=qwen_fused.attention_prefill_frontier.digest().as_hex(),
        variant_stable_id=stable_id,
      ),
    )
  }
  let wide_prefill = qwen_wide_query_prefill_variant(
    qwen_fused.attention_prefill_frontier,
  )
  let wide_prefill_root = arguments.absolute_new_output +
    "/reusable-qwen-prefill-wide-query-attention"
  match wide_prefill {
    Some(variant) =>
      write_new_candidate_pair(
        wide_prefill_root,
        variant.source().bytes(),
        qwen_compiler_attention_recipe(
          qwen_fused.first_operation + 3,
          "paged-attention-prefill-wide-query-functional-tile",
          set,
          variant,
          frontier_sha256=qwen_fused.attention_prefill_frontier
            .digest()
            .as_hex(),
          variant_stable_id=variant.functional().plan().candidate().stable_id(),
        ),
      )
    None => ()
  }
  let decode_root = arguments.absolute_new_output +
    "/reusable-qwen-decode-attention"
  write_new_candidate_pair(
    decode_root,
    qwen_fused.attention_decode.source().bytes(),
    qwen_compiler_attention_recipe(
      qwen_fused.first_operation + 3,
      "paged-attention-decode-functional-tile",
      set,
      qwen_fused.attention_decode,
      frontier_sha256=qwen_fused.attention_decode_frontier.digest().as_hex(),
      variant_stable_id=qwen_fused.attention_decode
        .functional()
        .plan()
        .candidate()
        .stable_id(),
    ),
  )
  publish_decode_frontier(arguments.absolute_new_output, set, qwen_fused) catch {
    _ => raise Publication
  }
  let decode_split_root = arguments.absolute_new_output +
    "/reusable-qwen-decode-split-attention"
  write_new_candidate_pair(
    decode_split_root,
    qwen_fused.attention_decode_split.source_bytes(),
    qwen_fused.attention_decode_split.recipe_bytes(),
  )
  println("schema=lunaflux-qwen3-bf16-candidate-export-cli.v1")
  println("recipe=dense_qwen3_bf16_paged_aot_v12")
  println(
    "weight_route_manifest_sha256=" + arguments.weight_route_manifest_sha256,
  )
  println("model_content_sha256=" + evidence.model_content_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("target=" + evidence.target())
  println("max_query_rows=" + arguments.max_query_rows.to_string())
  println("max_query_tokens=" + arguments.max_query_tokens.to_string())
  println("operation_count=" + evidence.operation_count().to_string())
  println("candidate_root=" + arguments.absolute_new_output + "/candidate-root")
  println(
    "candidate_inventory=" +
    arguments.absolute_new_output +
    "/candidate.files.sha256#sha256=" +
    evidence.inventory_sha256(),
  )
  println("candidate_set_sha256=" + evidence.candidate_set_sha256())
  println("export_record_sha256=" + evidence.export_record_sha256())
  println("compiler_invoked=0")
  println("device_opened=0")
  println("runtime_authority=0")
  println("physical_readiness=0")
  println("reusable_fused_residual_source=" + fused_root + "/kernel.cu")
  println("reusable_fused_residual_recipe=" + fused_root + "/kernel.recipe")
  println(
    "reusable_fused_residual_candidate_sha256=" +
    production.candidate_digest().as_hex(),
  )
  println(
    "reusable_fused_residual_first_operation=" +
    residual.residual_fallback().operation_id().as_int().to_string(),
  )
  println(
    "reusable_fused_residual_function_symbol=" + production.function_symbol(),
  )
  let residual_launch = production.launch_dimensions()
  println(
    "reusable_fused_residual_launch=" +
    residual_launch.grid_x().to_string() +
    "," +
    residual_launch.grid_y().to_string() +
    "," +
    residual_launch.block_x().to_string() +
    "," +
    residual_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_ingress_source=" + ingress_root + "/kernel.cu")
  println("reusable_qwen_ingress_recipe=" + ingress_root + "/kernel.recipe")
  println(
    "reusable_qwen_ingress_candidate_sha256=" +
    qwen_fused.postprocess.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_ingress_first_operation=" +
    qwen_fused.first_operation.to_string(),
  )
  println(
    "reusable_qwen_ingress_function_symbol=" +
    qwen_fused.postprocess.function_symbol(),
  )
  let ingress_launch = qwen_fused.postprocess.launch_dimensions()
  println(
    "reusable_qwen_ingress_launch=" +
    ingress_launch.grid_x().to_string() +
    "," +
    ingress_launch.grid_y().to_string() +
    "," +
    ingress_launch.block_x().to_string() +
    "," +
    ingress_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_readonly_source=" + readonly_root + "/kernel.cu")
  println("reusable_qwen_readonly_recipe=" + readonly_root + "/kernel.recipe")
  println(
    "reusable_qwen_readonly_candidate_sha256=" +
    qwen_fused.attention_read.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_readonly_operation=" +
    (qwen_fused.first_operation + 3).to_string(),
  )
  println(
    "reusable_qwen_readonly_function_symbol=" +
    qwen_fused.attention_read.function_symbol(),
  )
  let readonly_launch = qwen_fused.attention_read.launch_dimensions()
  println(
    "reusable_qwen_readonly_launch=" +
    readonly_launch.grid_x().to_string() +
    "," +
    readonly_launch.grid_y().to_string() +
    "," +
    readonly_launch.block_x().to_string() +
    "," +
    readonly_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_prefill_source=" + prefill_root + "/kernel.cu")
  println("reusable_qwen_prefill_recipe=" + prefill_root + "/kernel.recipe")
  println(
    "reusable_qwen_prefill_compilation_sha256=" +
    selected_prefill.digest().as_hex(),
  )
  println(
    "reusable_qwen_prefill_frontier_sha256=" +
    qwen_fused.attention_prefill_frontier.digest().as_hex(),
  )
  println(
    "reusable_qwen_prefill_frontier_variants=" +
    qwen_fused.attention_prefill_frontier.variants().length().to_string(),
  )
  for variant in qwen_fused.attention_prefill_frontier.variants() {
    let stable_id = variant.functional().plan().candidate().stable_id()
    println(
      "reusable_qwen_prefill_variant=" +
      stable_id.to_string() +
      "," +
      variant.source().symbol().to_owned() +
      "," +
      variant.digest().as_hex(),
    )
  }
  println(
    "reusable_qwen_prefill_partitioned_frontier_sha256=" +
    qwen_fused.attention_prefill_partitioned.digest().as_hex(),
  )
  println(
    "reusable_qwen_prefill_partitioned_frontier_variants=" +
    partitioned_prefill_variants.length().to_string(),
  )
  for index, variant in partitioned_prefill_variants {
    let compilation = variant.compilation()
    let partitions = compilation.functional().schedule().key_value_partitions()
    let (partial_symbol, merge_symbol) = compilation.source().symbols()
    println(
      "reusable_qwen_prefill_partitioned_variant=" +
      variant.parallel_workgroup_target().to_string() +
      "," +
      partitions.to_string() +
      "," +
      qwen_partitioned_prefill_variant_root(
        arguments.absolute_new_output,
        partitioned_prefill_variants,
        index,
      ) +
      "," +
      partial_symbol.to_owned() +
      "," +
      merge_symbol.to_owned() +
      "," +
      compilation.digest().as_hex(),
    )
  }
  match wide_prefill {
    Some(variant) => {
      let (query_rows, key_value_rows, _) = variant
        .functional()
        .plan()
        .candidate()
        .tile()
        .dimensions()
      println(
        "reusable_qwen_prefill_wide_source=" + wide_prefill_root + "/kernel.cu",
      )
      println(
        "reusable_qwen_prefill_wide_recipe=" +
        wide_prefill_root +
        "/kernel.recipe",
      )
      println(
        "reusable_qwen_prefill_wide_tile=" +
        query_rows.to_string() +
        "," +
        key_value_rows.to_string(),
      )
    }
    None => println("reusable_qwen_prefill_wide_source=absent")
  }
  let (prefill_grid_x, prefill_grid_y, prefill_block_x, prefill_shared) = selected_prefill
    .lowering()
    .launch()
  println(
    "reusable_qwen_prefill_launch=" +
    prefill_grid_x.to_string() +
    "," +
    prefill_grid_y.to_string() +
    "," +
    prefill_block_x.to_string() +
    "," +
    prefill_shared.to_string(),
  )
  println("reusable_qwen_decode_source=" + decode_root + "/kernel.cu")
  println("reusable_qwen_decode_recipe=" + decode_root + "/kernel.recipe")
  println(
    "reusable_qwen_decode_compilation_sha256=" +
    qwen_fused.attention_decode.digest().as_hex(),
  )
  let (decode_grid_x, decode_grid_y, decode_block_x, decode_shared) = qwen_fused.attention_decode
    .lowering()
    .launch()
  println(
    "reusable_qwen_decode_launch=" +
    decode_grid_x.to_string() +
    "," +
    decode_grid_y.to_string() +
    "," +
    decode_block_x.to_string() +
    "," +
    decode_shared.to_string(),
  )
  println(
    "reusable_qwen_decode_split_source=" + decode_split_root + "/kernel.cu",
  )
  println(
    "reusable_qwen_decode_split_recipe=" + decode_split_root + "/kernel.recipe",
  )
  println(
    "reusable_qwen_decode_split_candidate_sha256=" +
    qwen_fused.attention_decode_split.source_digest().as_hex(),
  )
  let (partial_symbol, merge_symbol) = qwen_fused.attention_decode_split.function_symbols()
  println("reusable_qwen_decode_split_partial_symbol=" + partial_symbol)
  println("reusable_qwen_decode_split_merge_symbol=" + merge_symbol)
  let (partial_launch, merge_launch) = qwen_fused.attention_decode_split.launch_dimensions()
  println(
    "reusable_qwen_decode_split_launches=" +
    partial_launch.grid_x().to_string() +
    "," +
    partial_launch.grid_y().to_string() +
    "," +
    partial_launch.grid_z().to_string() +
    ";" +
    merge_launch.grid_x().to_string() +
    "," +
    merge_launch.grid_y().to_string() +
    "," +
    merge_launch.grid_z().to_string(),
  )
}

///|
async fn main {
  try run_qwen3_candidate_export(@env.args()) catch {
    error => {
      println("outcome=failed")
      println("error=" + candidate_export_error_name(error))
      abort("Qwen3 BF16 candidate export failed")
    }
  } noraise {
    _ => ()
  }
}
