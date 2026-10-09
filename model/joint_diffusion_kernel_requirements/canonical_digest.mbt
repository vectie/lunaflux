///|
fn canonical_requirements_digest(
  identity : @spec.ModelIdentity,
  projection : @joint_diffusion_plan.JointDiffusionProjectionContract,
  entries : ArrayView[JointDiffusionKernelRequirement],
) -> JointDiffusionKernelRequirementsDigest raise JointDiffusionKernelRequirementsError {
  let writer = RequirementCanonicalWriter::new()
  writer.text("lunaflux/joint-diffusion-kernel-requirements/v10")
  writer.text(identity.content().as_hex())
  writer.text(identity.plan().as_hex())
  writer.byte(
    match projection.storage() {
      Float32InputWeightBiasBFloat16Output => 1
    },
  )
  writer.byte(
    match projection.output_storage() {
      Float32InputWeightBiasFloat32Output => 1
      BFloat16InputWeightBiasBFloat16Output => 2
    },
  )
  writer.byte(
    match projection.video_layout() {
      BatchFrameHeightWidthChannelTemporalHeightWidth => 1
    },
  )
  writer.byte(
    match projection.audio_layout() {
      ChannelMajorTime => 1
    },
  )
  writer.int(entries.length())
  for entry in entries {
    encode_operation(writer, entry.operation)
    writer.byte(capability_tag(entry.capability))
  }
  { hex: writer.hasher.finalize_hex(), }
}
