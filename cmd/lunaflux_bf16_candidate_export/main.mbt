///|
fn load_authenticated_single_row_plan(
  arguments : CandidateExportArguments,
) -> @plan.ModelPlan raise LunaBf16CandidateExportCliError {
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.model_root) catch {
    _ => raise ModelAdmission
  }
  let admitted = try
    @config_file.load(
      root,
      @approved_fs.ApprovedRelativeLocator::new(arguments.config_locator),
      @config_file.ModelConfigDigest::from_sha256(arguments.config_sha256),
      @spec.ContentDigest::from_sha256(arguments.model_content_sha256),
      @config_file.ModelConfigFileLimits::new(
        maximum_bytes=65536L,
        maximum_depth=16,
      ),
    )
  catch {
    _ => {
      root.close() catch {
        _ => ()
      }
      raise ModelAdmission
    }
  } noraise {
    value => value
  }
  root.close() catch {
    _ => raise ModelRootCleanup
  }
  @llama.build_paged(
    admitted,
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  ) catch {
    _ => raise ModelAdmission
  }
}

///|
fn prepare_candidate_set(
  plan : @plan.ModelPlan,
  arguments : CandidateExportArguments,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise LunaBf16CandidateExportCliError {
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=arguments.tokens_per_page,
      total_page_count=arguments.total_page_count,
      max_arena_bytes=1073741824L,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=plan.shape_constraints().max_batch_rows(),
    max_query_tokens=arguments.max_query_tokens,
    max_page_table_entries=arguments.max_page_table_entries,
  ) catch {
    _ => raise CandidatePreparation
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise CandidatePreparation
  }
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=arguments.toolchain_sha256,
      major=arguments.compiler_major.reinterpret_as_int(),
      minor=arguments.compiler_minor.reinterpret_as_int(),
      patch=arguments.compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  ) catch {
    _ => raise CandidatePreparation
  }
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=4096,
      max_source_bytes=16777216,
      max_recipe_bytes=16777216,
      max_total_bytes=1073741824,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
}

///|
async fn run_candidate_export(
  raw_arguments : ArrayView[String],
) -> Unit raise LunaBf16CandidateExportCliError {
  let arguments = parse_candidate_export_arguments(raw_arguments)
  let plan = load_authenticated_single_row_plan(arguments)
  let set = prepare_candidate_set(plan, arguments)
  let evidence = @luna_bf16_candidate_export.export_new(
    set,
    arguments.absolute_new_output,
  ) catch {
    _ => raise Publication
  }
  println("schema=lunaflux-bf16-candidate-export-cli.v1")
  println("model_content_sha256=" + evidence.model_content_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("target=" + evidence.target())
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
}

///|
async fn main {
  try! run_candidate_export(@env.args())
}
