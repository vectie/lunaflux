///|
const PLAN_MAX_CONFIG_BYTES : Int64 = 1048576L

///|
const PLAN_MAX_JSON_DEPTH : Int = 32

///|
const PLAN_MAX_KV_BUDGET_BYTES : Int64 = 1099511627776L

///|
const PLAN_MAX_TOKENS_PER_PAGE : Int = 4096

///|
const PLAN_MAX_PAGE_COUNT : Int64 = 2147483647L

///|
priv suberror PlanError {
  InvalidKvBudgetBytes
  InvalidTokensPerPage
}

///|
priv struct AdmittedPlanEvidence {
  plan : @plan.ModelPlan
  layout : @device_layout.DeviceKvLayout
  config_digest : @config_file.ModelConfigDigest
  kv_budget_bytes : Int64
  physical_page_bytes : Int64
  token_capacity : Int64
}

///|
fn parse_kv_budget(text : StringView) -> Int64 raise PlanError {
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise InvalidKvBudgetBytes
  }
  if value <= 0L || value > PLAN_MAX_KV_BUDGET_BYTES {
    raise InvalidKvBudgetBytes
  }
  value
}

///|
fn parse_tokens_per_page(text : StringView) -> Int raise PlanError {
  let value : Int = @strconv.from_str(text) catch {
    _ => raise InvalidTokensPerPage
  }
  if value <= 0 || value > PLAN_MAX_TOKENS_PER_PAGE {
    raise InvalidTokensPerPage
  }
  value
}

///|
fn plan_config_limits() -> @config_file.ModelConfigFileLimits raise {
  @config_file.ModelConfigFileLimits::new(
    maximum_bytes=PLAN_MAX_CONFIG_BYTES,
    maximum_depth=PLAN_MAX_JSON_DEPTH,
  )
}

///|
fn derive_kv_layout(
  plan : @plan.ModelPlan,
  kv_budget_bytes : Int64,
  tokens_per_page : Int,
) -> (@device_layout.DeviceKvLayout, Int64) raise {
  let one_page = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=1,
      max_arena_bytes=PLAN_MAX_KV_BUDGET_BYTES,
    ),
  )
  let physical_page_bytes = one_page.arena_bytes()
  let page_count = kv_budget_bytes / physical_page_bytes
  if page_count <= 0L || page_count > PLAN_MAX_PAGE_COUNT {
    raise InvalidKvBudgetBytes
  }
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=page_count.to_int(),
      max_arena_bytes=kv_budget_bytes,
    ),
  )
  (layout, physical_page_bytes)
}

///|
fn admit_plan_evidence(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  config_digest : @config_file.ModelConfigDigest,
  model_digest : @spec.ContentDigest,
  kv_budget_bytes : Int64,
  tokens_per_page : Int,
) -> AdmittedPlanEvidence raise {
  let metadata = @config_file.load(
    root,
    locator,
    config_digest,
    model_digest,
    plan_config_limits(),
  )
  let plan = @llama.build_paged(metadata)
  let (layout, physical_page_bytes) = derive_kv_layout(
    plan, kv_budget_bytes, tokens_per_page,
  )
  {
    plan,
    layout,
    config_digest,
    kv_budget_bytes,
    physical_page_bytes,
    token_capacity: layout.total_page_count().to_int64() *
    layout.tokens_per_page().to_int64(),
  }
}

///|
fn capability_name(capability : @plan.KernelCapabilityId) -> String {
  if capability == @plan.KernelCapabilityId::embedding_lookup() {
    "embedding_lookup"
  } else if capability == @plan.KernelCapabilityId::rms_norm() {
    "rms_norm"
  } else if capability == @plan.KernelCapabilityId::qkv_projection() {
    "qkv_projection"
  } else if capability ==
    @plan.KernelCapabilityId::positioned_rotary_embedding() {
    "positioned_rotary_embedding"
  } else if capability == @plan.KernelCapabilityId::paged_causal_attention() {
    "paged_causal_attention"
  } else if capability == @plan.KernelCapabilityId::dense_projection() {
    "dense_projection"
  } else if capability == @plan.KernelCapabilityId::residual_add() {
    "residual_add"
  } else if capability == @plan.KernelCapabilityId::gated_mlp() {
    "gated_mlp"
  } else {
    "capability_\{capability.as_int()}"
  }
}

///|
fn append_report_line(output : StringBuilder, line : String) -> Unit {
  output.write_string(line)
  output.write_string("\n")
}

///|
fn render_plan_evidence(evidence : AdmittedPlanEvidence) -> String {
  let plan = evidence.plan
  let shape = plan.shape_constraints()
  let layout = evidence.layout
  let full_context_pages = (
      shape.max_sequence_tokens().to_int64() +
      layout.tokens_per_page().to_int64() -
      1L
    ) /
    layout.tokens_per_page().to_int64()
  let full_context_capacity = layout.total_page_count().to_int64() /
    full_context_pages
  let output = StringBuilder()
  append_report_line(output, "LunaFlux admitted startup plan")
  append_report_line(
    output,
    "model config sha256: \{evidence.config_digest.as_hex()}; evidence: authenticated snapshot",
  )
  append_report_line(
    output,
    "model content sha256: \{plan.identity().content().as_hex()}; evidence: caller-declared, not weight-verified",
  )
  append_report_line(
    output,
    "model plan sha256: \{plan.identity().plan().as_hex()}; evidence: derived from authenticated config",
  )
  append_report_line(
    output, "architecture: dense Llama; reason: authenticated bounded config",
  )
  append_report_line(output, "execution semantics: paged KV")
  append_report_line(output, "operation count: \{plan.operations().length()}")
  append_report_line(output, "maximum batch rows: \{shape.max_batch_rows()}")
  append_report_line(
    output,
    "maximum sequence tokens: \{shape.max_sequence_tokens()}",
  )
  append_report_line(output, "vocabulary size: \{shape.vocabulary_size()}")
  append_report_line(output, "hidden size: \{shape.hidden_size()}")
  append_report_line(output, "intermediate size: \{shape.intermediate_size()}")
  append_report_line(output, "attention heads: \{shape.attention_head_count()}")
  append_report_line(output, "KV heads: \{shape.kv_head_count()}")
  append_report_line(output, "head dimension: \{shape.head_dimension()}")
  append_report_line(
    output,
    "semantic workspace ceiling bytes: \{plan.workspace_bounds().max_temporary_bytes()}",
  )
  append_report_line(
    output,
    "required kernel capabilities: \{plan.required_capabilities().length()}",
  )
  for index, capability in plan.required_capabilities() {
    append_report_line(
      output,
      "  capability[\{index}]: \{capability_name(capability)} (id \{capability.as_int()}); reason: required by admitted model plan",
    )
  }
  append_report_line(output, "KV budget bytes: \{evidence.kv_budget_bytes}")
  append_report_line(
    output,
    "KV physical page bytes: \{evidence.physical_page_bytes}",
  )
  append_report_line(output, "KV tokens per page: \{layout.tokens_per_page()}")
  append_report_line(
    output,
    "KV physical page count: \{layout.total_page_count()}",
  )
  append_report_line(output, "KV planned arena bytes: \{layout.arena_bytes()}")
  append_report_line(output, "KV token capacity: \{evidence.token_capacity}")
  append_report_line(
    output,
    "full-context sequence capacity: \{full_context_capacity}",
  )
  append_report_line(
    output, "KV decision: maximum whole-page capacity within explicit budget; unused tail bytes remain outside the planned arena",
  )
  append_report_line(
    output, "kernel catalog: unresolved; required capabilities are known, but no digest-pinned execution manifest was admitted",
  )
  append_report_line(output, "model materialization: not performed")
  append_report_line(output, "device allocation: not performed")
  append_report_line(
    output, "readiness: false; kernel and physical-device gates remain open",
  )
  output.to_string()
}

///|
/// Authenticates one caller-supplied model configuration and returns its
/// deterministic semantic and KV-capacity report without materializing
/// weights or opening a device. The approved root is closed before the report
/// is rendered or returned.
pub fn render(
  root_path : String,
  config_locator : String,
  config_sha : String,
  model_sha : String,
  kv_budget_text : String,
  tokens_per_page_text : String,
) -> String raise {
  let locator = @approved_fs.ApprovedRelativeLocator::new(config_locator)
  let config_digest = @config_file.ModelConfigDigest::from_sha256(config_sha)
  let model_digest = @spec.ContentDigest::from_sha256(model_sha)
  let kv_budget_bytes = parse_kv_budget(kv_budget_text)
  let tokens_per_page = parse_tokens_per_page(tokens_per_page_text)
  let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
  let evidence = admit_plan_evidence(
    root, locator, config_digest, model_digest, kv_budget_bytes, tokens_per_page,
  ) catch {
    error => {
      root.close() catch {
        _ => ()
      }
      raise error
    }
  }
  root.close()
  render_plan_evidence(evidence)
}

///|
/// Prints the deterministic report returned by `render`.
pub fn run(
  root_path : String,
  config_locator : String,
  config_sha : String,
  model_sha : String,
  kv_budget_text : String,
  tokens_per_page_text : String,
) -> Unit raise {
  println(
    render(
      root_path, config_locator, config_sha, model_sha, kv_budget_text, tokens_per_page_text,
    )
    .trim_end()
    .to_owned(),
  )
}
