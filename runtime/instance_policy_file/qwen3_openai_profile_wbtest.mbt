///|
fn qwen3_openai_policy_fixture() -> String {
  openai_v3_policy_fixture()
  .replace(
    old="\"maximum_credential_bytes\":4096,\"max_head_bytes\":4096,\"max_body_bytes\":4096",
    new="\"maximum_credential_bytes\":128,\"max_head_bytes\":65536,\"max_body_bytes\":1048576",
  )
  .replace(
    old="\"http_step_work_units\":17,\"response_step_work_units\":31,\"inbound_step_work_units\":17,\"outbound_step_work_units\":31,\"max_messages\":8",
    new="\"http_step_work_units\":256,\"response_step_work_units\":256,\"inbound_step_work_units\":256,\"outbound_step_work_units\":256,\"max_messages\":32",
  )
  .replace(
    old="\"system_prefix\":\"\",\"system_suffix\":\"\",\"user_prefix\":\"\",\"user_suffix\":\"\",\"assistant_prefix\":\"\",\"assistant_suffix\":\"\",\"assistant_cue\":\"\"",
    new="\"system_prefix\":\"<|im_start|>system\\n\",\"system_suffix\":\"<|im_end|>\\n\",\"user_prefix\":\"<|im_start|>user\\n\",\"user_suffix\":\"<|im_end|>\\n\",\"assistant_prefix\":\"<|im_start|>assistant\\n\",\"assistant_suffix\":\"<|im_end|>\\n\",\"assistant_cue\":\"<|im_start|>assistant\\n\"",
  )
  .replace(
    old="\"model_alias\":\"tiny-bf16\"",
    new="\"model_alias\":\"qwen3-0.6b-bf16\"",
  )
  .replace(
    old="\"response_id_prefix\":\"resp_lunaflux_\"",
    new="\"response_id_prefix\":\"resp_lunaflux_qwen3_\"",
  )
  .replace(
    old="\"cache_scope_ascii\":\"openai-v3\"",
    new="\"cache_scope_ascii\":\"qwen3-openai-v1\"",
  )
  .replace(
    old="\"max_new_tokens\":8,\"context_ceiling\":32,\"sampling_seed\":7",
    new="\"max_new_tokens\":256,\"context_ceiling\":40960,\"sampling_seed\":1",
  )
}

///|
test "Qwen3 OpenAI profile binds exact ChatML and service claims" {
  let policy = try! parse_policy_fixture(qwen3_openai_policy_fixture())
  assert_eq(policy.external_protocol().mode(), OpenAiResponsesV1)
  assert_eq(policy.external_protocol().invocation_path(), "/v1/responses")
  guard policy.openai_service() is Some(openai) else {
    abort("Qwen3 OpenAI service claims were not retained")
  }
  assert_eq(openai.maximum_credential_bytes(), 128)
  assert_eq(openai.model_alias(), "qwen3-0.6b-bf16")
  assert_eq(openai.system_prefix(), "<|im_start|>system\n")
  assert_eq(openai.system_suffix(), "<|im_end|>\n")
  assert_eq(openai.user_prefix(), "<|im_start|>user\n")
  assert_eq(openai.user_suffix(), "<|im_end|>\n")
  assert_eq(openai.assistant_prefix(), "<|im_start|>assistant\n")
  assert_eq(openai.assistant_suffix(), "<|im_end|>\n")
  assert_eq(openai.assistant_cue(), "<|im_start|>assistant\n")
  assert_eq(openai.response_id_prefix(), "resp_lunaflux_qwen3_")
  assert_eq(openai.cache_scope_ascii(), "qwen3-openai-v1")
  assert_eq(openai.max_new_tokens(), 256)
  assert_eq(openai.context_ceiling(), 40960)
  assert_eq(openai.sampling_seed(), 1UL)
}

///|
test "Qwen3 OpenAI profile rejects an ambient credential field" {
  let hostile = qwen3_openai_policy_fixture().replace(
    old="\"maximum_credential_bytes\":128",
    new="\"maximum_credential_bytes\":128,\"credential\":\"forbidden\"",
  )
  try parse_policy_fixture(hostile) catch {
    UnknownField(area=OpenAiService, ..) => ()
    _ => abort("unexpected hostile Qwen3 OpenAI policy refusal")
  } noraise {
    _ => abort("hostile Qwen3 OpenAI policy was admitted")
  }
}
