///|
async test "canonical v1 admits one immutable full-graph paged execution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-manifest")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
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
      fixture_step_limits(),
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
async test "v1 rejects foreign weight content and multi-row specialization" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-envelope")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
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
        fixture_step_limits(),
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
        fixture_step_limits(rows=2),
        fixture_limits(
          plan.operations().length(),
          entry_points=plan.operations().length() + 1,
        ),
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected multi-row failure")
    } noraise {
      _ => fail("multi-row v1 specialization was admitted")
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
    old="\"schema_version\":1",
    new="\"schema_version\":1,\"schema_version\":1",
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
