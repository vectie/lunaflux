///|
fn materialized_test_bundle_limits() -> @deployment_bundle.DeploymentBundleLimits raise {
  @deployment_bundle.DeploymentBundleLimits::new(
    maximum_bytes=65536,
    maximum_base_image_bytes=512,
    maximum_locator_bytes=256,
  )
}

///|
fn materialized_test_evidence(
  launch_sha : String,
  descriptor_sha : String,
  policy_sha : String,
  worker_sha : String,
  descriptor_relative? : String = "runtime.json",
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux.deployment-bundle.v1\n" +
    "base_image=registry.invalid/lunaflux/cuda@sha256=".replace(
      old="sha256=",
      new="sha256:",
    ) +
    "a".repeat(64) +
    "\nlinux_architecture=x86_64\n" +
    "launch_file_sha256=\{launch_sha}\n" +
    "runtime_descriptor_relative=\{descriptor_relative}\n" +
    "runtime_descriptor_sha256=\{descriptor_sha}\n" +
    "instance_policy_relative=missing-policy.json\n" +
    "instance_policy_sha256=\{policy_sha}\n" +
    "lunaflux_executable_sha256=" +
    "1".repeat(64) +
    "\nworker_executable_sha256=\{worker_sha}\n" +
    "model_inventory_sha256=" +
    "2".repeat(64) +
    "\npolicy_inventory_sha256=" +
    "3".repeat(64) +
    "\nkernel_manifest_relative=execution.json\n" +
    "kernel_manifest_sha256=" +
    "4".repeat(64) +
    "\nkernel_inventory_sha256=" +
    "5".repeat(64) +
    "\nruntime_library_inventory_sha256=" +
    "6".repeat(64) +
    "\nrootfs_inventory_sha256=" +
    "7".repeat(64) +
    "\nassembly_input_sha256=" +
    "8".repeat(64) +
    "\nassembler_sha256=" +
    "9".repeat(64) +
    "\nexact_inventory=1\ncompiler_jit_free=1\n",
  )
}

///|
async fn write_materialized_test_bundle(
  root : String,
  evidence_descriptor_relative? : String = "runtime.json",
) -> Bytes {
  @fs.mkdir("\{root}/evidence", recursive=true)
  @fs.mkdir("\{root}/launch-root", recursive=true)
  @fs.mkdir("\{root}/model-root", recursive=true)
  @fs.mkdir("\{root}/policy-root", recursive=true)
  @fs.mkdir("\{root}/oci-context/rootfs/opt/lunaflux/kernels", recursive=true)
  @fs.mkdir("\{root}/oci-context/rootfs/opt/lunaflux/bin", recursive=true)
  let descriptor = @utf8.encode(legacy_descriptor_at_i8_route())
  let descriptor_sha = release_test_sha256(descriptor)
  @fs.write_file(
    "\{root}/model-root/runtime.json",
    descriptor,
    create_mode=CreateNew,
  )
  let worker = b"materialized worker"
  let worker_sha = release_test_sha256(worker)
  @fs.write_file(
    "\{root}/oci-context/rootfs/opt/lunaflux/bin/lunaflux-device-worker",
    worker,
    create_mode=CreateNew,
  )
  let policy_sha = "4".repeat(64)
  let launch = @utf8.encode(
    "{\"schema\":\"lunaflux.launch.v1\",\"model_root\":\"/var/lib/lunaflux/model\",\"kernel_root\":\"/opt/lunaflux/kernels\",\"policy_root\":\"/var/lib/lunaflux/policy\",\"runtime_descriptor\":{\"locator\":\"runtime.json\",\"sha256\":\"\{descriptor_sha}\"},\"instance_policy\":{\"locator\":\"missing-policy.json\",\"sha256\":\"\{policy_sha}\"},\"worker_executable\":{\"path\":\"/opt/lunaflux/bin/lunaflux-device-worker\",\"sha256\":\"\{worker_sha}\"},\"luna_approval\":{\"mode\":\"none\"}}",
  )
  let launch_sha = release_test_sha256(launch)
  @fs.write_file(
    "\{root}/launch-root/lunaflux.launch.json",
    launch,
    create_mode=CreateNew,
  )
  let evidence = materialized_test_evidence(
    launch_sha,
    descriptor_sha,
    policy_sha,
    worker_sha,
    descriptor_relative=evidence_descriptor_relative,
  )
  @fs.write_file(
    "\{root}/evidence/deployment-bundle.v1",
    evidence,
    create_mode=CreateNew,
  )
  evidence
}

///|
async test "materialized release reads staged descriptor for final target labels" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-materialized-release")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let evidence = write_materialized_test_bundle(root)
    try
      @runtime_instance.preflight_materialized_release(
        "\{root}#sha256=\{release_test_sha256(evidence)}",
        release_test_limits(),
        materialized_test_bundle_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(DescriptorAdmission) => ()
      _ => abort("mapped bundle failed before its first model descendant")
    } noraise {
      _ => abort("incomplete materialized bundle produced release evidence")
    }
  }
}

///|
async test "materialized release rejects bundle and launch substitution first" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-materialized-hostile")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let evidence = write_materialized_test_bundle(
      root,
      evidence_descriptor_relative="substituted.json",
    )
    try
      @runtime_instance.preflight_materialized_release(
        "x",
        release_test_limits(),
        materialized_test_bundle_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(DeploymentBundleAdmission) => ()
      _ => abort("short bundle argument failed at the wrong stage")
    } noraise {
      _ => abort("short bundle argument was admitted")
    }
    try
      @runtime_instance.preflight_materialized_release(
        "\{root}#sha256=\{"f".repeat(64)}",
        release_test_limits(),
        materialized_test_bundle_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(DeploymentBundleAdmission) => ()
      _ => abort("bundle digest substitution failed at the wrong stage")
    } noraise {
      _ => abort("bundle digest substitution was admitted")
    }
    try
      @runtime_instance.preflight_materialized_release(
        "\{root}#sha256=\{release_test_sha256(evidence)}",
        release_test_limits(),
        materialized_test_bundle_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(MaterializationBinding) => ()
      _ => abort("bundle/launch substitution failed at the wrong stage")
    } noraise {
      _ => abort("bundle/launch substitution was admitted")
    }
  }
}
