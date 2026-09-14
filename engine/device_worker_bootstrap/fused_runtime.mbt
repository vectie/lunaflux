///|
fn read_fused_runtime_snapshot(
  root : @approved_fs.ApprovedRoot,
  source : @worker_wire.WorkerBootstrapFusedRuntimeSource,
  maximum_bytes : Int64,
) -> Bytes raise DeviceWorkerBootstrapError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(source.locator()) catch {
    _ => raise Invalid(Locator)
  }
  let file = root.open_file(locator) catch { _ => raise Invalid(Locator) }
  let outcome : Result[Bytes, DeviceWorkerBootstrapError] = Ok(
    file.read_immutable_snapshot(maximum_bytes~) catch {
      _ => raise Invalid(RecipeLimit)
    },
  ) catch {
    error => Err(error)
  }
  let close_result : Result[Unit, DeviceWorkerBootstrapError] = Ok(
    file.close() catch {
      _ => raise Invalid(Locator)
    },
  ) catch {
    error => Err(error)
  }
  let bytes = match outcome {
    Ok(value) => value
    Err(error) => raise error
  }
  match close_result {
    Ok(_) => bytes
    Err(error) => raise error
  }
}

///|
/// Loads the optional descriptor-pinned source once while the inherited kernel
/// root is still open. Absence deliberately preserves the standalone plan.
priv struct LoadedOptionalFusedRuntime {
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog?
  bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle?
  production : @luna_paged_composite_artifact_admission.DigestPinnedFusedProductionRuntime?
  reusable_residual : @luna_fused_artifact_admission.ApprovedReusableFusedResidualRuntime?
  reusable_bundle : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle?
}

///|
priv enum FusedRuntimeSourceSchema {
  ReusableBundle
  ReusableResidualV1
  ProductionV1
  LegacyCatalogV1
}

///|
fn classify_fused_runtime_source(text : String) -> FusedRuntimeSourceSchema {
  if text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v3\n") ||
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v4\n") ||
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v5\n") ||
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v6\n") {
    ReusableBundle
  } else if text.has_prefix(
      "schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1\n",
    ) {
    ReusableResidualV1
  } else if text.has_prefix(
      "schema=lunaflux-fused-production-runtime-source.v1\n",
    ) {
    ProductionV1
  } else {
    LegacyCatalogV1
  }
}

///|
fn reject_reusable_qwen_bundle_for_non_qwen(
  runtime : LoadedOptionalFusedRuntime?,
) -> Unit raise DeviceWorkerBootstrapError {
  match runtime {
    Some(loaded) if loaded.reusable_bundle is Some(_) => raise Invalid(Recipe)
    _ => ()
  }
}

///|
fn fused_runtime_snapshot_limit(
  module_bytes : Int64,
) -> Int64 raise DeviceWorkerBootstrapError {
  // V6 permits one MiB of route observations, encoded as two hex characters
  // per byte, in addition to the pre-existing module/metadata envelope.
  let overhead = 2097152L + 65536L
  if module_bytes <= 0L || module_bytes > (0x7fffffffffffffffL - overhead) / 4L {
    raise Invalid(RecipeLimit)
  }
  module_bytes * 4L + overhead
}

///|
fn load_optional_fused_runtime(
  source : @worker_wire.EncodedBootstrapSource,
  kernel_root : @approved_fs.ApprovedRoot,
) -> LoadedOptionalFusedRuntime? raise DeviceWorkerBootstrapError {
  guard source.kernel().fused_runtime_source() is Some(fused) else {
    return None
  }
  let module_bytes = source.ceilings().max_total_module_bytes()
  let canonical = read_fused_runtime_snapshot(
    kernel_root,
    fused,
    fused_runtime_snapshot_limit(module_bytes),
  )
  let text = @utf8.decode(canonical) catch { _ => raise Invalid(Digest) }
  match classify_fused_runtime_source(text) {
    ReusableBundle => {
      let bundle = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
        canonical,
        fused.digest_sha256(),
      ) catch {
        error => raise FusedArtifact(error)
      }
      Some({
        catalog: None,
        bundle: None,
        production: None,
        reusable_residual: None,
        reusable_bundle: Some(bundle),
      })
    }
    ReusableResidualV1 => {
      let residual = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_residual_runtime(
        canonical,
        fused.digest_sha256(),
      ) catch {
        error => raise FusedArtifact(error)
      }
      Some({
        catalog: None,
        bundle: None,
        production: None,
        reusable_residual: Some(residual),
        reusable_bundle: None,
      })
    }
    ProductionV1 => {
      let production = @luna_paged_composite_artifact_admission.admit_digest_pinned_fused_production_runtime(
        canonical,
        fused.digest_sha256(),
      ) catch {
        _ => raise Invalid(Digest)
      }
      Some({
        catalog: Some(production.fused_catalog()),
        bundle: Some(production.fused_bundle()),
        production: Some(production),
        reusable_residual: None,
        reusable_bundle: None,
      })
    }
    LegacyCatalogV1 => {
      let pair = @luna_fused_artifact_admission.admit_digest_pinned_fused_runtime_source(
        canonical,
        fused.digest_sha256(),
      ) catch {
        error => raise FusedArtifact(error)
      }
      Some({
        catalog: Some(pair.0),
        bundle: Some(pair.1),
        production: None,
        reusable_residual: None,
        reusable_bundle: None,
      })
    }
  }
}
