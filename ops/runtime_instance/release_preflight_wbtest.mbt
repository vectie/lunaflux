///|
test "release evidence is canonical and root-free" {
  let (evidence, digest) = encode_release_evidence(
    "dense_llama_paged_aot_v5",
    "a".repeat(64),
    "b".repeat(64),
    "c".repeat(64),
    "d".repeat(64),
    "e".repeat(64),
    "f".repeat(64),
    "1".repeat(64),
    "2".repeat(64),
    "3".repeat(64),
    3,
    9,
    0,
  )
  inspect(
    evidence,
    content=(
      #|schema=lunaflux-release-preflight.v1
      #|runtime_recipe=dense_llama_paged_aot_v5
      #|launch_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      #|runtime_descriptor_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
      #|instance_policy_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      #|tokenizer_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
      #|worker_executable_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
      #|model_content_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
      #|model_plan_sha256=1111111111111111111111111111111111111111111111111111111111111111
      #|bootstrap_sha256=2222222222222222222222222222222222222222222222222222222222222222
      #|bootstrap_source_sha256=3333333333333333333333333333333333333333333333333333333333333333
      #|device_ordinal=3
      #|compute_major=9
      #|compute_minor=0
      #|semantic_join=1
      #|filesystem_authority_closed=1
      #|device_opened=0
      #|compiler_jit_authority=0
      #|
    ),
  )
  assert_eq(
    digest, "220287cda61dc79ddc806faabeb89ef1b6e59107e9cb2743901ef0a1e4e507ca",
  )
  assert_false(evidence.contains("/"))
  assert_false(evidence.contains("nvrtc"))
}

///|
test "tensor parallel preflight emits genuine group template digest" {
  let (evidence, _) = encode_release_evidence(
    "dense_llama_tensor_parallel_paged_aot_approved_v8",
    "a".repeat(64),
    "b".repeat(64),
    "c".repeat(64),
    "d".repeat(64),
    "e".repeat(64),
    "f".repeat(64),
    "1".repeat(64),
    "2".repeat(64),
    "3".repeat(64),
    0,
    tensor_parallel_group_template_sha256=Some("4".repeat(64)),
    8,
    0,
  )
  assert_true(
    evidence.contains("tensor_parallel_group_template_sha256=" + "4".repeat(64)),
  )
  assert_false(evidence.contains("bootstrap_sha256=" + "0".repeat(64)))
}
