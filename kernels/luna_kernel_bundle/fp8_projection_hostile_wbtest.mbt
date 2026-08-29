///|
fn fp8_hostile_artifact_entries(
  omit? : Int = -1,
  duplicate? : Int = -1,
) -> Array[@artifact.KernelEntryPointInput] {
  let result : Array[@artifact.KernelEntryPointInput] = []
  for index in 0..<4 {
    if index != omit {
      let entry = @artifact.KernelEntryPointInput::new(
        entry_point=fp8_bundle_entry(index),
        function_symbol=if index == 1 {
          "lunaflux_luna_dense_projection_fp8_e4m3_w8a8_release_v1"
        } else {
          "fixture_kernel_" + index.to_string()
        },
      )
      result.push(entry)
      if index == duplicate {
        result.push(entry)
      }
    }
  }
  result
}

///|
fn fp8_hostile_artifact_limits() -> @artifact.KernelArtifactLimits {
  try! @artifact.KernelArtifactLimits::new(
    max_modules=2,
    max_entry_points=8,
    max_module_bytes=1024L,
    max_total_module_bytes=2048L,
    max_function_symbol_bytes=96,
  )
}

///|
test "FP8 catalog-v4 artifacts reject omitted and duplicate entry points" {
  let plan = fp8_bundle_plan()
  let contracts = fp8_bundle_contracts(plan)
  let modules = [
    @artifact.KernelModuleInput::from_bytes(
      digest=fp8_bundle_family(0).module_digest(),
      module_bytes=fp8_bundle_bytes(),
    ),
  ]
  try
    @artifact.admit_paged_v4(
      contracts,
      modules,
      fp8_hostile_artifact_entries(omit=1),
      fp8_hostile_artifact_limits(),
    )
  catch {
    MissingEntryPoint(_) => ()
    _ => fail("unexpected omitted-entry rejection")
  } noraise {
    _ => fail("omitted FP8 entry point was admitted")
  }
  try
    @artifact.admit_paged_v4(
      contracts,
      modules,
      fp8_hostile_artifact_entries(duplicate=1),
      fp8_hostile_artifact_limits(),
    )
  catch {
    DuplicateEntryPoint(_) => ()
    _ => fail("unexpected duplicate-entry rejection")
  } noraise {
    _ => fail("duplicate FP8 entry point was admitted")
  }
}
