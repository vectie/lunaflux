///|
test "complete token-hash artifact binding fails closed at Query-B projection" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1'),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2'),
    ("deepseek-ai/DeepSeek-V4-Pro", '3'),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4'),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5'),
  ]
  for entry in cases {
    let (model_id, fill) = entry
    let requirements = operand_plan_test_requirements(model_id~, fill~)
    let candidate = operand_plan_test_candidate(requirements)
    assert_artifact_binding_test_query_b_operation_unavailable(
      requirements, candidate,
    )
  }
}
