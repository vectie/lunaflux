///|
async test "canonical v2 admits exact multi-row full-graph paged execution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-manifest")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let batch = @llama.LlamaPagedBatchEnvelope::new(2)
    let plan = @llama.build_paged(metadata, batch~)
    let weights = fixture_weight_layout(metadata)
    assert_true(weights.identity().plan() != plan.identity().plan())
    assert_eq(weights.identity().content(), plan.identity().content())
    let target = fixture_target()
    let manifest = fixture_manifest(plan, target)
    write_fixture_file(directory, "execution.json", @utf8.encode(manifest))
    write_fixture_file(directory, "kernel.cubin", fixture_module_bytes())
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let admission = load_paged(
      root,
      @approved_fs.ApprovedRelativeLocator::new("execution.json"),
      fixture_digest(manifest),
      plan,
      weights,
      fixture_layout(plan),
      target,
      fixture_step_limits(rows=2),
      fixture_limits(plan.operations().length()),
    ) catch {
      AdmissionRejected(Catalog) => abort("catalog admission failed")
      AdmissionRejected(StaticPlan) => abort("static plan admission failed")
      AdmissionRejected(MemoryPlan) => abort("memory plan admission failed")
      AdmissionRejected(LaunchContracts) => abort("contract admission failed")
      AdmissionRejected(ArtifactSource) =>
        abort("artifact source admission failed")
      AdmissionRejected(ArtifactFiles) =>
        abort("artifact file admission failed")
      AdmissionRejected(Blueprint) => abort("blueprint admission failed")
      _ => abort("execution manifest admission failed")
    }
    assert_eq(admission.model_plan(), plan)
    assert_eq(admission.weight_layout(), weights)
    assert_eq(admission.blueprint().scope(), FullGraph)
    assert_eq(
      admission.blueprint().steps().length(),
      plan.operations().length(),
    )
    assert_eq(admission.artifacts().module_count(), 1)
    assert_eq(
      admission.contracts().contracts().length(),
      plan.operations().length(),
    )
    assert_eq(admission.artifacts().entry_point_count(), 9)
    assert_false(root.is_closed())
    root.close()
  }
}

///|
fn v3_manifest_claim(
  valid : String,
  operation_id : Int,
  specialization_locator? : String = "luna/residual.specialization",
) -> String {
  valid.replace(
    old="\"schema_version\":2",
    new="\"schema_version\":3,\"luna_authorization\":{" +
      "\"operation_id\":" +
      operation_id.to_string() +
      "," +
      "\"specialization_locator\":\"" +
      specialization_locator +
      "\"," +
      "\"specialization_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"," +
      "\"capability_locator\":\"luna/residual.capability\"," +
      "\"capability_sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"," +
      "\"signature_algorithm\":\"external_ed25519_v1\"," +
      "\"key_identity_sha256\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\"," +
      "\"signature_sha256\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\"}",
  )
}

///|
fn first_residual_operation_id(plan : @plan.ModelPlan) -> Int {
  for operation in plan.operations() {
    if operation.kind() == ResidualAdd {
      return operation.id().as_int()
    }
  }
  abort("fixture has no residual operation")
}

///|
fn v3_approval(
  manifest_digest? : String = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
) -> @worker_wire.WorkerBootstrapLunaApproval raise {
  @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
    manifest_digest_sha256=manifest_digest,
    key_identity_sha256="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    signature_digest_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    approved_source_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  )
}

///|
async test "v3 rejects mismatched approval before ordinary module open" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-luna-binding")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
    let target = fixture_target()
    let specialized = v3_manifest_claim(
      fixture_manifest(plan, target),
      first_residual_operation_id(plan),
    )
    write_fixture_file(directory, "v3.json", @utf8.encode(specialized))
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("v3.json"),
        fixture_digest(specialized),
        plan,
        fixture_weight_layout(metadata),
        fixture_layout(plan),
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
        luna_approval=Some(
          v3_approval(
            manifest_digest="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
          ),
        ),
      )
    catch {
      AdmissionRejected(LunaExternalApproval) => ()
      AdmissionRejected(ArtifactFiles) =>
        fail("module opened before Luna approval binding")
      _ => fail("unexpected Luna approval-binding failure")
    } noraise {
      _ => fail("mismatched Luna approval was admitted")
    }
    root.close()
  }
}

///|
async test "v3 requires independent approval and v2 rejects unused approval" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-luna-approval")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
    let target = fixture_target()
    let ordinary = fixture_manifest(plan, target)
    let specialized = v3_manifest_claim(
      ordinary,
      first_residual_operation_id(plan),
    )
    write_fixture_file(directory, "v2.json", @utf8.encode(ordinary))
    write_fixture_file(directory, "v3.json", @utf8.encode(specialized))
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("v3.json"),
        fixture_digest(specialized),
        plan,
        fixture_weight_layout(metadata),
        fixture_layout(plan),
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
      )
    catch {
      AdmissionRejected(LunaExternalApproval) => ()
      _ => fail("unexpected missing Luna approval failure")
    } noraise {
      _ => fail("v3 admitted without independent approval")
    }
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("v2.json"),
        fixture_digest(ordinary),
        plan,
        fixture_weight_layout(metadata),
        fixture_layout(plan),
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
        luna_approval=Some(v3_approval()),
      )
    catch {
      AdmissionRejected(LunaExternalApproval) => ()
      _ => fail("unexpected unused Luna approval failure")
    } noraise {
      _ => fail("v2 consumed an unused Luna approval")
    }
    root.close()
  }
}

///|
async test "v3 validates both evidence locators before artifact open" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-luna-locator")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
    let target = fixture_target()
    let invalid = v3_manifest_claim(
      fixture_manifest(plan, target),
      first_residual_operation_id(plan),
      specialization_locator="../residual.specialization",
    )
    write_fixture_file(directory, "v3.json", @utf8.encode(invalid))
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("v3.json"),
        fixture_digest(invalid),
        plan,
        fixture_weight_layout(metadata),
        fixture_layout(plan),
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
        luna_approval=Some(v3_approval()),
      )
    catch {
      AdmissionRejected(LunaSpecialization) => ()
      AdmissionRejected(ArtifactFiles) =>
        fail("artifact opened before Luna locator admission")
      _ => fail("unexpected Luna locator failure")
    } noraise {
      _ => fail("ambient Luna locator was admitted")
    }
    root.close()
  }
}

///|
async test "v2 rejects foreign weight content and row-envelope mismatch" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-envelope")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let batch = @llama.LlamaPagedBatchEnvelope::new(2)
    let plan = @llama.build_paged(metadata, batch~)
    let target = fixture_target()
    let manifest = fixture_manifest(plan, target)
    write_fixture_file(directory, "execution.json", @utf8.encode(manifest))
    write_fixture_file(directory, "kernel.cubin", fixture_module_bytes())
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let locator = @approved_fs.ApprovedRelativeLocator::new("execution.json")
    let digest = fixture_digest(manifest)
    try
      load_paged(
        root,
        locator,
        digest,
        plan,
        fixture_weight_layout(fixture_metadata('1')),
        fixture_layout(plan),
        target,
        fixture_step_limits(rows=2),
        fixture_limits(plan.operations().length()),
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected foreign-content failure")
    } noraise {
      _ => fail("foreign weight content was admitted")
    }
    try
      load_paged(
        root,
        locator,
        digest,
        plan,
        fixture_weight_layout(metadata),
        fixture_layout(plan),
        target,
        fixture_step_limits(),
        fixture_limits(
          plan.operations().length(),
          entry_points=plan.operations().length() + 1,
        ),
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected row-envelope failure")
    } noraise {
      _ => fail("mismatched row envelope was admitted")
    }
    root.close()
  }
}

///|
test "schema rejects duplicate, unknown, and reordered declarations" {
  let metadata = fixture_metadata('0')
  let plan = @llama.build_paged(metadata)
  let target = fixture_target()
  let limits = fixture_limits(plan.operations().length())
  let valid = fixture_manifest(plan, target)
  let duplicate = valid.replace(
    old="\"schema_version\":2",
    new="\"schema_version\":2,\"schema_version\":2",
  )
  try parse_manifest(@utf8.encode(duplicate), plan, target, limits) catch {
    DuplicateField(_) => ()
    _ => fail("unexpected duplicate-field failure")
  } noraise {
    _ => fail("duplicate field was admitted")
  }
  let unknown = valid.replace(
    old="\"profile\":{\"id\":1}",
    new="\"profile\":{\"id\":1,\"extra\":0}",
  )
  try parse_manifest(@utf8.encode(unknown), plan, target, limits) catch {
    UnknownField(area=Profile, index=_) => ()
    _ => fail("unexpected unknown-field failure")
  } noraise {
    _ => fail("unknown field was admitted")
  }
  let reordered = valid.replace(
    old="\"operation_id\":0",
    new="\"operation_id\":1",
  )
  try parse_manifest(@utf8.encode(reordered), plan, target, limits) catch {
    InvalidField(area=Operation, issue=InvalidOrder, index=0) => ()
    _ => fail("unexpected order failure")
  } noraise {
    _ => fail("reordered operation was admitted")
  }
}

///|
async test "invalid or unreferenced exports fail before module open" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-source-order")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
    let weights = fixture_weight_layout(metadata)
    let layout = fixture_layout(plan)
    let target = fixture_target()
    let valid = fixture_manifest(plan, target)
    let invalid_symbol = valid.replace(
      old="\"symbol\":\"kernel_1\"",
      new="\"symbol\":\"1bad\"",
    )
    let unreferenced = valid.replace(
      old="]}],\"operations\"",
      new=",{\"family_id\":100,\"entry_point_id\":100,\"symbol\":\"extra\"}]}],\"operations\"",
    )
    write_fixture_file(directory, "invalid.json", @utf8.encode(invalid_symbol))
    write_fixture_file(
      directory,
      "unreferenced.json",
      @utf8.encode(unreferenced),
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("invalid.json"),
        fixture_digest(invalid_symbol),
        plan,
        weights,
        layout,
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
      )
    catch {
      AdmissionRejected(ArtifactSource) => ()
      AdmissionRejected(ArtifactFiles) =>
        fail("module opened before symbol admission")
      _ => fail("unexpected invalid-symbol ordering failure")
    } noraise {
      _ => fail("invalid symbol was admitted")
    }
    try
      load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new("unreferenced.json"),
        fixture_digest(unreferenced),
        plan,
        weights,
        layout,
        target,
        fixture_step_limits(),
        fixture_limits(plan.operations().length()),
      )
    catch {
      InvalidField(area=Module, issue=UnsupportedValue, index=0) => ()
      InvalidField(area=Module, issue=InvalidOrder, index=_) =>
        abort("unreferenced fixture broke ordering")
      AdmissionRejected(ArtifactSource) =>
        abort("unreferenced fixture reached artifact source")
      AdmissionRejected(ArtifactFiles) =>
        fail("module opened before linkage admission")
      _ => fail("unexpected unreferenced-export ordering failure")
    } noraise {
      _ => fail("unreferenced export was admitted")
    }
    root.close()
  }
}
