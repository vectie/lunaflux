///|
/// Source-only diagnostic through the production compiler; no artifact
/// admission or deployment receipt is created by this fixture.
fn main raise {
  let activation = @plan.ActivationRef::new(0)
  let output = @plan.ActivationRef::new(1)
  let tensors = [0, 1, 2].map(i => @plan.TensorRef::new(i))
  let operation = @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(1),
    kind=GatedMlp,
    inputs=[
      Activation(activation),
      Tensor(tensors[0]),
      Tensor(tensors[1]),
      Tensor(tensors[2]),
    ],
    outputs=[output],
    shape=GatedMlpShape(hidden_size=1024, intermediate_size=3072),
    required_capability=@plan.KernelCapabilityId::gated_mlp(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=8192,
  )
  let operands = [
    @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationInput(activation),
      byte_count=2048L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[0]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[1]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[2]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationOutput(output),
      byte_count=2048L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=Workspace,
      byte_count=2048L * 3072L * 2L,
      alignment=16L,
    ),
  ]
  let compiled = @luna_cuda_projection_aot.lower_reusable_projection_cuda_aot_candidate(
    operation,
    profile,
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    @luna_specialization.LunaCompilerPolicy::new(
      identity=@luna_specialization.LunaCompilerIdentity::new(
        toolchain_digest_sha256="0".repeat(64),
        major=13,
        minor=1,
        patch=115,
      ),
      optimization_level=3,
      fused_multiply_add=false,
      reassociate=false,
      max_registers=128,
    ),
    operands,
    @luna_cuda_projection_aot.ProjectionAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  )
  println(@utf8.decode(compiled.source_bytes()))
  println("// RECIPE-BEGIN")
  for line in @utf8.decode(compiled.recipe_bytes()).split("\n") {
    println("// \{line}")
  }
}
