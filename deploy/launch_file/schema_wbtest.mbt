///|
fn launch_fixture(approval? : String = "{\"mode\":\"none\"}") -> String {
  "{\"schema\":\"lunaflux.launch.v1\",\"model_root\":\"/opt/lunaflux/model\",\"kernel_root\":\"/opt/lunaflux/kernels\",\"policy_root\":\"/opt/lunaflux/policy\",\"runtime_descriptor\":{\"locator\":\"runtime.json\",\"sha256\":\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"instance_policy\":{\"locator\":\"instance.json\",\"sha256\":\"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"worker_executable\":{\"path\":\"/opt/lunaflux/bin/worker\",\"sha256\":\"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"luna_approval\":\{approval}}"
}

///|
fn launch_v2_fixture(
  recipe? : String = "dense_llama_i8_paged_aot_v6",
  approval? : String = "{\"mode\":\"none\"}",
) -> String {
  launch_fixture(approval~).replace(
    old="\"schema\":\"lunaflux.launch.v1\"",
    new="\"schema\":\"lunaflux.launch.v2\",\"runtime_recipe\":\"\{recipe}\"",
  )
}

///|
fn launch_v3_tensor_parallel_fixture(
  approval? : String = "{\"mode\":\"external_ed25519_v1\",\"capability_manifest_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"key_identity_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"signature_sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"approved_source_sha256\":\"7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
) -> String {
  launch_fixture(approval~).replace(
    old="\"schema\":\"lunaflux.launch.v1\"",
    new="\"schema\":\"lunaflux.launch.v3\",\"runtime_recipe\":\"dense_llama_tensor_parallel_paged_aot_approved_v8\"",
  )
}

///|
fn launch_v4_fp8_fixture(
  approval? : String = "{\"mode\":\"external_ed25519_v1\",\"capability_manifest_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"key_identity_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"signature_sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"approved_source_sha256\":\"7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
) -> String {
  launch_fixture(approval~).replace(
    old="\"schema\":\"lunaflux.launch.v1\"",
    new="\"schema\":\"lunaflux.launch.v4\",\"runtime_recipe\":\"dense_llama_fp8_reusable_paged_aot_approved_v9\"",
  )
}

///|
fn launch_v5_qwen3_fixture(
  approval? : String = "{\"mode\":\"none\"}",
) -> String {
  launch_fixture(approval~).replace(
    old="\"schema\":\"lunaflux.launch.v1\"",
    new="\"schema\":\"lunaflux.launch.v5\",\"runtime_recipe\":\"dense_qwen3_bf16_paged_aot_v12\"",
  )
}

///|
fn parse_fixture(text : String) -> LaunchFileAdmission raise LaunchFileError {
  let digest = LaunchFileDigest::from_sha256(
    "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  parse_launch(
    @utf8.encode(text),
    digest,
    LaunchFileLimits::new(maximum_bytes=65536L, maximum_json_depth=8),
  )
}

///|
test "launch schema admits none and exact external approval" {
  let none = try! parse_fixture(launch_fixture())
  assert_true(none.luna_approval() is None)
  assert_eq(none.runtime_recipe(), DenseLlamaPagedAotV5)
  let external = try! parse_fixture(
    launch_fixture(
      approval="{\"mode\":\"external_ed25519_v1\",\"capability_manifest_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"key_identity_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"signature_sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"approved_source_sha256\":\"7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
    ),
  )
  guard external.luna_approval() is Some(approval) else {
    fail("external approval was discarded")
  }
  assert_eq(approval.signature_sha256()[0], '6')
}

///|
test "launch v2 authenticates one exact closed runtime recipe" {
  let i8 = try! parse_fixture(launch_v2_fixture())
  assert_eq(i8.runtime_recipe(), DenseLlamaI8PagedAotV6)
  let legacy = try! parse_fixture(
    launch_v2_fixture(recipe="dense_llama_paged_aot_v5"),
  )
  assert_eq(legacy.runtime_recipe(), DenseLlamaPagedAotV5)
  let mistral = try! parse_fixture(
    launch_v2_fixture(recipe="dense_mistral_bf16_paged_aot_v7"),
  )
  assert_eq(mistral.runtime_recipe(), DenseMistralBf16PagedAotV7)
}

///|
test "launch v5 admits only unapproved dense Qwen3 BF16 v12" {
  let qwen = try! parse_fixture(launch_v5_qwen3_fixture())
  assert_eq(qwen.runtime_recipe(), DenseQwen3Bf16PagedAotV12)
  for
    substituted in [
      "dense_llama_i8_paged_aot_v6", "dense_mistral_bf16_paged_aot_v7",
    ] {
    try
      parse_fixture(
        launch_v5_qwen3_fixture().replace(
          old="dense_qwen3_bf16_paged_aot_v12",
          new=substituted,
        ),
      )
    catch {
      InvalidField(Root) => ()
      _ => fail("Qwen3 recipe substitution reported an unrelated failure")
    } noraise {
      _ => fail("Qwen3 launch admitted another model-family recipe")
    }
  }
  try
    parse_fixture(
      launch_v5_qwen3_fixture(
        approval="{\"mode\":\"external_ed25519_v1\",\"capability_manifest_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"key_identity_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"signature_sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"approved_source_sha256\":\"7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
      ),
    )
  catch {
    InvalidField(LunaApproval) => ()
    _ => fail("Qwen3 approval substitution reported an unrelated failure")
  } noraise {
    _ => fail("Qwen3 v12 silently discarded an external approval")
  }
}

///|
test "launch v1 v2 recipe and approval cross decoding fails closed" {
  let external = "{\"mode\":\"external_ed25519_v1\",\"capability_manifest_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"key_identity_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"signature_sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"approved_source_sha256\":\"7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}"
  let hostile = [
    launch_fixture().replace(
      old="\"model_root\":",
      new="\"runtime_recipe\":\"dense_llama_i8_paged_aot_v6\",\"model_root\":",
    ),
    launch_v2_fixture().replace(
      old="\"runtime_recipe\":\"dense_llama_i8_paged_aot_v6\",",
      new="",
    ),
    launch_v2_fixture(recipe="dense_llama_i8_paged_aot_v7"),
    launch_v2_fixture(approval=external),
    launch_v2_fixture(
      recipe="dense_mistral_bf16_paged_aot_v7",
      approval=external,
    ),
  ]
  for text in hostile {
    try parse_fixture(text) catch {
      UnknownField(Root)
      | MissingField(Root)
      | InvalidField(Root)
      | InvalidField(LunaApproval) => ()
      _ => fail("unexpected cross-schema launch failure")
    } noraise {
      _ => fail("cross-schema launch envelope was admitted")
    }
  }
}

///|
test "launch v3 admits only approved tensor parallel v8" {
  let admitted = try! parse_fixture(launch_v3_tensor_parallel_fixture())
  assert_eq(
    admitted.runtime_recipe(),
    DenseLlamaTensorParallelPagedAotApprovedV8,
  )
  assert_true(admitted.luna_approval() is Some(_))
  let hostile = [
    launch_v3_tensor_parallel_fixture(approval="{\"mode\":\"none\"}"),
    launch_v3_tensor_parallel_fixture().replace(
      old="dense_llama_tensor_parallel_paged_aot_approved_v8",
      new="dense_llama_tensor_parallel_paged_aot_v5",
    ),
    launch_v2_fixture(
      recipe="dense_llama_tensor_parallel_paged_aot_approved_v8",
    ),
  ]
  for text in hostile {
    try parse_fixture(text) catch {
      MissingField(LunaApproval) | InvalidField(Root) => ()
      _ => fail("unexpected tensor-parallel launch rejection")
    } noraise {
      _ => fail("tensor-parallel launch downgrade was admitted")
    }
  }
}

///|
test "launch v4 admits only externally approved reusable FP8 v9" {
  let admitted = try! parse_fixture(launch_v4_fp8_fixture())
  assert_eq(admitted.runtime_recipe(), DenseLlamaFp8ReusablePagedAotApprovedV9)
  assert_true(admitted.luna_approval() is Some(_))
  for
    hostile in [
      launch_v4_fp8_fixture(approval="{\"mode\":\"none\"}"),
      launch_v4_fp8_fixture().replace(
        old="dense_llama_fp8_reusable_paged_aot_approved_v9",
        new="dense_llama_i8_paged_aot_v6",
      ),
      launch_v3_tensor_parallel_fixture().replace(
        old="dense_llama_tensor_parallel_paged_aot_approved_v8",
        new="dense_llama_fp8_reusable_paged_aot_approved_v9",
      ),
    ] {
    try parse_fixture(hostile) catch {
      MissingField(LunaApproval) | InvalidField(Root) => ()
      _ => fail("unexpected reusable FP8 launch rejection")
    } noraise {
      _ => fail("reusable FP8 launch downgrade was admitted")
    }
  }
}

///|
test "launch schema rejects duplicates unknowns paths and approval drift" {
  let canonical = launch_fixture()
  let hostile = [
    canonical.replace(
      old="\"schema\":\"lunaflux.launch.v1\"",
      new="\"schema\":\"lunaflux.launch.v1\",\"schema\":\"lunaflux.launch.v1\"",
    ),
    canonical.replace(
      old="\"model_root\":",
      new="\"ambient\":0,\"model_root\":",
    ),
    canonical.replace(old="/opt/lunaflux/model", new="/opt/../model"),
    canonical.replace(
      old="{\"mode\":\"none\"}",
      new="{\"mode\":\"none\",\"signature_sha256\":\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
    ),
  ]
  for text in hostile {
    try parse_fixture(text) catch {
      DuplicateField | UnknownField(_) | InvalidField(_) => ()
      _ => fail("unexpected hostile launch failure")
    } noraise {
      _ => fail("hostile launch envelope was admitted")
    }
  }
}

///|
test "argument grammar is literal and digest is independent" {
  let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  let (root, parsed) = try! parse_argument("/opt/lunaflux#sha256=\{digest}")
  assert_eq(root, "/opt/lunaflux")
  assert_eq(parsed.as_hex(), digest)
  for
    hostile in [
      "/opt/lunaflux",
      "/opt/lunaflux#sha256=ABC",
      "relative#sha256=\{digest}",
      "/opt/lunaflux/#sha256=\{digest}",
    ] {
    try parse_argument(hostile) catch {
      InvalidArgument | InvalidExpectedDigest => ()
      _ => fail("unexpected argument failure")
    } noraise {
      _ => fail("hostile MODEL grammar was admitted")
    }
  }
}
