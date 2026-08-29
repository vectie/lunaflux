///|
/// Rejects every scalar and lexical I8 descriptor claim before descendant I/O.
///
/// This is deliberately pure: it opens no root, file, device, or allocation.
/// Later admission repeats the focused semantic constructors while composing
/// the exact model, numeric file, and execution evidence.
fn preflight_i8_claims(
  claims : I8RuntimeDescriptorClaims,
) -> Unit raise RuntimeDescriptorFileError {
  if claims.kernels.policy != I8_RUNTIME_POLICY {
    raise AdmissionRejected(I8ProductionPolicy)
  }
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(claims.model.config_locator) catch {
      _ => raise AdmissionRejected(ModelConfig)
    },
  )
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(
      claims.model.numeric_weights_locator,
    ) catch {
      _ => raise AdmissionRejected(NumericWeightInspection)
    },
  )
  if claims.model.config_locator == claims.model.numeric_weights_locator {
    raise AdmissionRejected(NumericWeightInspection)
  }
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(claims.kernels.manifest_locator) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
  )
  ignore(
    @config_file.ModelConfigDigest::from_sha256(claims.model.config_digest) catch {
      _ => raise AdmissionRejected(ModelConfig)
    },
  )
  ignore(
    @spec.ContentDigest::from_sha256(claims.model.content_digest) catch {
      _ => raise AdmissionRejected(ModelIdentity)
    },
  )
  ignore(
    @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      claims.model.numeric_weight_artifact_digest,
    ) catch {
      _ => raise AdmissionRejected(NumericWeightInspection)
    },
  )
  ignore(
    @execution_manifest_file.ExecutionManifestDigest::from_sha256(
      claims.kernels.manifest_digest,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
  )
  ignore(
    @worker_wire.WorkerBootstrapDigest::from_sha256(
      claims.kernels.bootstrap_v3_digest,
    ) catch {
      _ => raise AdmissionRejected(BootstrapDigest)
    },
  )
  ignore(admit_ceilings(claims.ceilings))
  ignore(admit_i8_device_capability(claims.execution))
  ignore(admit_worker_limits(claims.worker))
  ignore(admit_inference_limits(claims.inference))
  if claims.model.max_batch_rows != claims.worker.max_plan_rows {
    raise AdmissionRejected(ModelPlan)
  }
  ignore(
    @llama.LlamaPagedBatchEnvelope::new(claims.model.max_batch_rows) catch {
      _ => raise AdmissionRejected(ModelPlan)
    },
  )
}
