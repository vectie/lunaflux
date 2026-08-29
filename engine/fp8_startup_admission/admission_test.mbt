///|
fn startup_limits(
  canonical_bytes? : Int = 4096,
) -> @fp8_startup_admission.Fp8StartupAdmissionLimits {
  try! @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
    max_canonical_bytes=canonical_bytes,
  )
}

///|
test "exact numeric, device, kernel, launch, and artifact evidence joins" {
  let fixture = startup_fixture()
  let admitted = @fp8_startup_admission.admit(
    fixture.plan,
    fixture.device_numeric,
    fixture.kernels,
    startup_limits(),
  )
  assert_eq(
    admitted.version(),
    @fp8_startup_admission.Fp8StartupAdmissionVersion::v1(),
  )
  assert_eq(admitted.model_plan().identity(), fixture.plan.identity())
  assert_eq(
    admitted.device_numeric_capability().digest(),
    fixture.device_numeric.digest(),
  )
  assert_eq(
    admitted.kernel_artifacts().manifest().digest(),
    fixture.kernels.manifest().digest(),
  )
  assert_eq(admitted.kernel_artifacts().artifacts().module_count(), 1)
  assert_eq(admitted.digest().as_hex().length(), 64)
  inspect(
    admitted.digest().as_hex(),
    content="a62431540658a06a56508975d219c5af7f8d608a91d90eb88ea31fcb32d8090b",
  )
  inspect(
    @crypto.bytes_to_hex_string(admitted.canonical_bytes()),
    content="6c756e61666c75782e6670382d737461727475702d61646d697373696f6e2e7631000100000000000000400000000000000031313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131400000000000000065333537643465636537353330376633343138323637393461376534336531343830343339633233353765326566633432326464633662636462666565613538400000000000000037303537313666323964376337393733623039373761353765356461653365393135313439343935376232393437363339616533636264313232343730356665400000000000000063643436666435306563366163306133383165656231356432626633316534316566306536356633623763346663383465343565626134646465356561393031400000000000000034353935636464663330646562666662323636666535316136663934373235393235313936363366616536313666356263386237666132346563386339656537",
  )
  let repeated = startup_fixture()
  assert_eq(
    @fp8_startup_admission.admit(
      repeated.plan,
      repeated.device_numeric,
      repeated.kernels,
      startup_limits(),
    ).digest(),
    admitted.digest(),
  )
}

///|
test "substituted observed device evidence is rejected before publication" {
  let fixture = startup_fixture()
  try
    @fp8_startup_admission.admit(
      fixture.plan,
      startup_device(major=9, minor=0),
      fixture.kernels,
      startup_limits(),
    )
  catch {
    Fp8StartupRejected(DeviceEvidence) => ()
    _ => fail("unexpected device substitution failure")
  } noraise {
    _ => fail("substituted device evidence was published")
  }
}

///|
test "foreign model plan and undersized canonical envelope fail closed" {
  let fixture = startup_fixture()
  let foreign = fp8_foreign_plan()
  try
    @fp8_startup_admission.admit(
      foreign,
      fixture.device_numeric,
      fixture.kernels,
      startup_limits(),
    )
  catch {
    Fp8StartupRejected(ModelPlan) => ()
    _ => fail("unexpected foreign-plan failure")
  } noraise {
    _ => fail("foreign plan was joined to FP8 startup evidence")
  }
  try
    @fp8_startup_admission.admit(
      fixture.plan,
      fixture.device_numeric,
      fixture.kernels,
      startup_limits(canonical_bytes=1),
    )
  catch {
    Fp8StartupRejected(CanonicalBytes) => ()
    _ => fail("unexpected canonical bound failure")
  } noraise {
    _ => fail("oversized startup canonical bytes were published")
  }
}

///|
fn fp8_foreign_plan() -> @plan.ModelPlan {
  let base = startup_plan()
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations=base.operations(),
    final_output=base.final_output(),
    numeric_schema=@plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
      base.operations(),
      [@plan.OperationId::new(1)],
    ),
  )
}
