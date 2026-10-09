///|
test "attention-related executable operand ABIs fail closed" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let wrong_attention_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_attention_operands=true,
  )
  try
    admit(
      requirements,
      target,
      wrong_attention_operands,
      artifact_test_limits(),
    )
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, 14)
    _ => fail("unexpected attention-output operand error")
  } noraise {
    _ => fail("wrong attention-output weight size admitted")
  }
  let wrong_head_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_head_operands=true,
  )
  try
    admit(requirements, target, wrong_head_operands, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, 28)
    _ => fail("unexpected language-model-head operand error")
  } noraise {
    _ => fail("hidden-output role admitted as logits output")
  }
  let wrong_pooling_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_pooling_operands=true,
  )
  try
    admit(requirements, target, wrong_pooling_operands, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, 12)
    _ => fail("unexpected DSA pooling operand error")
  } noraise {
    _ => fail("truncated DSA raw-index output admitted")
  }
  let wrong_dsa_index_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_dsa_index_operands=true,
  )
  try
    admit(
      requirements,
      target,
      wrong_dsa_index_operands,
      artifact_test_limits(),
    )
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, 11)
    _ => fail("unexpected DSA index-selection operand error")
  } noraise {
    _ => fail("truncated DSA selected-index output admitted")
  }
  let wrong_sparse_attention_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_sparse_attention_operands=true,
  )
  let mut sparse_ordinal = -1
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaSparseAttention(_, _, _, _) {
      sparse_ordinal = ordinal
    }
  }
  try
    admit(
      requirements,
      target,
      wrong_sparse_attention_operands,
      artifact_test_limits(),
    )
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, sparse_ordinal)
    _ => fail("unexpected sparse-attention operand error")
  } noraise {
    _ => fail("truncated sparse-attention output admitted")
  }
}
