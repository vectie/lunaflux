///|
fn load_tp_fixture_rank(
  root : @approved_fs.ApprovedRoot,
  source_name : String,
  digest : ExecutionManifestDigest,
  fixture : TensorParallelManifestFixture,
  rank : Int,
  inspection_rank? : Int = rank,
  binding_rank? : Int = rank,
  generation? : UInt64 = 7UL,
) -> TensorParallelExecutionAdmission raise {
  let operation_count = fixture.model.operations().length()
  let limits = fixture_limits(operation_count)
  load_tensor_parallel(
    root,
    @approved_fs.ApprovedRelativeLocator::new(source_name),
    digest,
    fixture.model,
    @worker_protocol.ModelPlanGeneration::from_wire(generation),
    fixture.inspections[rank].rank_plan(),
    fixture.inspections[inspection_rank],
    fixture.devices[rank],
    fixture.kv_plan.rank_layout(rank),
    fixture.kv_plan.rank_binding(binding_rank),
    fixture.target,
    fixture_step_limits(rows=2),
    limits,
    tp_manifest_launch_limits(operation_count),
    tp_manifest_execution_limits(
      operation_count,
      limits.artifact_file().artifact(),
    ),
  )
}

///|
async test "tensor-parallel v2 admits exact rank-local execution plan" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-manifest-admission")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let weights = tp_manifest_weight_file()
    write_fixture_file(directory, "weights.safetensors", weights)
    write_fixture_file(directory, "kernel.cubin", fixture_module_bytes())
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let fixture = tp_manifest_fixture(root, weights)
    let manifest = fixture_manifest(fixture.model, fixture.target)
    write_fixture_file(directory, "execution.json", @utf8.encode(manifest))
    let admission = load_tp_fixture_rank(
      root,
      "execution.json",
      fixture_digest(manifest),
      fixture,
      0,
    ) catch {
      AdmissionRejected(Catalog) => abort("TP catalog admission failed")
      AdmissionRejected(TensorParallelLaunchContracts) =>
        abort("TP launch-contract admission failed")
      AdmissionRejected(ArtifactSource) => abort("TP artifact source failed")
      AdmissionRejected(ArtifactFiles) => abort("TP artifact load failed")
      AdmissionRejected(TensorParallelExecutionPlan) =>
        abort("TP execution-plan admission failed")
      _ => abort("TP manifest admission failed")
    }
    assert_eq(admission.model_plan(), fixture.model)
    assert_eq(admission.model_generation().value(), 7UL)
    assert_eq(admission.rank_plan(), fixture.inspections[0].rank_plan())
    assert_eq(admission.device_plan(), fixture.devices[0])
    assert_eq(admission.local_kv_layout(), fixture.kv_plan.rank_layout(0))
    assert_eq(admission.local_kv_binding(), fixture.kv_plan.rank_binding(0))
    assert_eq(admission.profile().id().as_int(), 1)
    assert_eq(admission.contracts().rank(), 0)
    assert_eq(admission.contracts().world_size(), 2)
    assert_eq(admission.execution_plan().rank(), 0)
    assert_eq(admission.execution_plan().device_ordinal(), 2)
    assert_eq(
      admission.execution_plan().operation_count(),
      fixture.model.operations().length(),
    )
    assert_eq(admission.artifacts().module_count(), 1)
    assert_false(root.is_closed())
    root.close()
  }
}

///|
async test "tensor-parallel manifest rejects digest manifest and rank substitution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-manifest-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let weights = tp_manifest_weight_file()
    write_fixture_file(directory, "weights.safetensors", weights)
    write_fixture_file(directory, "kernel.cubin", fixture_module_bytes())
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let fixture = tp_manifest_fixture(root, weights)
    let valid = fixture_manifest(fixture.model, fixture.target)
    let foreign = valid.replace(
      old=fixture.model.identity().plan().as_hex(),
      new="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    write_fixture_file(directory, "execution.json", @utf8.encode(valid))
    write_fixture_file(directory, "foreign.json", @utf8.encode(foreign))
    try
      load_tp_fixture_rank(
        root,
        "execution.json",
        ExecutionManifestDigest::from_sha256(
          "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        ),
        fixture,
        0,
      )
    catch {
      DigestMismatch => ()
      _ => fail("unexpected TP digest-substitution failure")
    } noraise {
      _ => fail("foreign TP manifest digest was admitted")
    }
    try
      load_tp_fixture_rank(
        root,
        "foreign.json",
        fixture_digest(foreign),
        fixture,
        0,
      )
    catch {
      IdentityMismatch(PlanDigest) => ()
      _ => fail("unexpected TP semantic-manifest substitution failure")
    } noraise {
      _ => fail("foreign TP semantic manifest was admitted")
    }
    try
      load_tp_fixture_rank(
        root,
        "execution.json",
        fixture_digest(valid),
        fixture,
        0,
        inspection_rank=1,
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected TP shard-rank substitution failure")
    } noraise {
      _ => fail("foreign TP sharded inspection was admitted")
    }
    try
      load_tp_fixture_rank(
        root,
        "execution.json",
        fixture_digest(valid),
        fixture,
        0,
        binding_rank=1,
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected TP KV-rank substitution failure")
    } noraise {
      _ => fail("foreign TP KV rank binding was admitted")
    }
    try
      load_tp_fixture_rank(
        root,
        "execution.json",
        fixture_digest(valid),
        fixture,
        0,
        generation=8UL,
      )
    catch {
      UnsupportedModelEnvelope => ()
      _ => fail("unexpected TP model-generation substitution failure")
    } noraise {
      _ => fail("generation-unbound TP KV binding was admitted")
    }
    root.close()
  }
}

///|
async test "tensor-parallel manifest rejects Luna v3 without exact specialization" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-manifest-luna")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let weights = tp_manifest_weight_file()
    write_fixture_file(directory, "weights.safetensors", weights)
    write_fixture_file(directory, "kernel.cubin", fixture_module_bytes())
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let fixture = tp_manifest_fixture(root, weights)
    let specialized = v3_manifest_claim(
      fixture_manifest(fixture.model, fixture.target),
      first_residual_operation_id(fixture.model),
    )
    write_fixture_file(directory, "execution.json", @utf8.encode(specialized))
    try
      load_tp_fixture_rank(
        root,
        "execution.json",
        fixture_digest(specialized),
        fixture,
        0,
      )
    catch {
      AdmissionRejected(TensorParallelSpecialization) => ()
      AdmissionRejected(ArtifactFiles) =>
        fail("TP Luna v3 reached ordinary artifact loading")
      _ => fail("unexpected TP Luna-v3 rejection")
    } noraise {
      _ => fail("TP Luna v3 was admitted without an exact specialization")
    }
    root.close()
  }
}
