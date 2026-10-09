///|
pub(all) suberror Glm53MoeExpertBankPlanError {
  IdentityMismatch
  ManifestDigestMismatch
  RequirementsDigestMismatch
  LayerMismatch
  ProfileMismatch
  OperationMismatch
  OperandMismatch
  RecipeMismatch
} derive(Debug, Eq)

///|
pub struct Glm53MoeExpertBankCandidatePlan {
  priv execution_identity : @spec.ModelIdentity
  priv manifest_layout_digest : @spec.PlanDigest
  priv layer_index : Int
  priv expert_bank_layout_digest : @glm53_weights.Glm53MoeExpertBankLayoutDigest
  priv routed_recipe_digest : @glm_hybrid_cuda_aot.GlmHybridCudaRecipeDigest
  priv shared_recipe_digest : @glm_hybrid_cuda_aot.GlmHybridCudaRecipeDigest
  priv digest : Glm53MoeExpertBankCandidatePlanDigest
}

///|
pub struct Glm53MoeExpertBankCandidatePlanDigest(String) derive(Debug, Eq, Hash)

///|
fn exact_operand(
  operand : @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  role : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole,
  bytes : Int64,
) -> Bool {
  operand.role() == role &&
  operand.byte_count() == bytes &&
  operand.alignment() == 16L
}

///|
/// Joins one exact GLM manifest-derived expert-bank layout to the two
/// correctness candidates which consume it. This is startup evidence only.
pub fn join_moe_expert_bank_candidates(
  manifest : @glm53_weights.Glm53TensorManifest,
  bank_layout : @glm53_weights.Glm53MoeExpertBankLayout,
  routed : @glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate,
  shared : @glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate,
  expected_layer_index~ : Int,
) -> Glm53MoeExpertBankCandidatePlan raise Glm53MoeExpertBankPlanError {
  if bank_layout.layer_index() != expected_layer_index {
    raise LayerMismatch
  }
  if bank_layout.execution_identity() != manifest.execution_identity() ||
    routed.source_identity() != manifest.execution_identity() ||
    shared.source_identity() != manifest.execution_identity() {
    raise IdentityMismatch
  }
  if bank_layout.manifest_layout_digest() != manifest.layout_digest() {
    raise ManifestDigestMismatch
  }
  if routed.requirements_digest() != shared.requirements_digest() {
    raise RequirementsDigestMismatch
  }
  let (expert_count, hidden_width, routed_family, shared_family) = match
    manifest.profile() {
    FullText =>
      (
        256,
        6144,
        @glm_hybrid_cuda_aot.FullRoutedExperts256,
        @glm_hybrid_cuda_aot.FullSharedExpert,
      )
    FlashMultimodal => (288, 4096, FlashRoutedExperts288, FlashSharedExpert)
  }
  if bank_layout.routed_expert_count() != expert_count ||
    bank_layout.hidden_width() != hidden_width ||
    bank_layout.intermediate_width() != 2048 ||
    routed.family() != routed_family ||
    shared.family() != shared_family {
    raise ProfileMismatch
  }
  guard routed.operation() is RoutedExpertFeedForward(actual_experts, 2048, _) else {
    raise OperationMismatch
  }
  guard shared.operation() is SharedExpertFeedForward(1, 2048, _) else {
    raise OperationMismatch
  }
  if actual_experts != expert_count ||
    routed.operands().length() != 8 ||
    shared.operands().length() != 6 {
    raise OperationMismatch
  }
  if !exact_operand(
      routed.operands()[4],
      MoeExpertGateWeights,
      bank_layout.routed_gate().byte_count(),
    ) ||
    !exact_operand(
      routed.operands()[5],
      MoeExpertUpWeights,
      bank_layout.routed_up().byte_count(),
    ) ||
    !exact_operand(
      routed.operands()[6],
      MoeExpertDownWeights,
      bank_layout.routed_down().byte_count(),
    ) ||
    !exact_operand(
      shared.operands()[2],
      MoeSharedGateWeights,
      bank_layout.shared_gate().byte_count(),
    ) ||
    !exact_operand(
      shared.operands()[3],
      MoeSharedUpWeights,
      bank_layout.shared_up().byte_count(),
    ) ||
    !exact_operand(
      shared.operands()[4],
      MoeSharedDownWeights,
      bank_layout.shared_down().byte_count(),
    ) {
    raise OperandMismatch
  }
  let routed_recipe = @utf8.decode(routed.recipe_bytes()) catch {
    _ => raise RecipeMismatch
  }
  let shared_recipe = @utf8.decode(shared.recipe_bytes()) catch {
    _ => raise RecipeMismatch
  }
  if !routed_recipe.contains("stage=routed-selected-expert-") ||
    !shared_recipe.contains("stage=shared-expert-") ||
    !routed_recipe.contains(
      "qualification=serial-correctness-source-only-non-bindable",
    ) ||
    !shared_recipe.contains(
      "qualification=serial-correctness-source-only-non-bindable",
    ) {
    raise RecipeMismatch
  }
  let digest_text = "lunaflux.glm53-moe-expert-bank-candidate-plan.v1|" +
    manifest.execution_identity().content().as_hex() +
    "|" +
    manifest.execution_identity().plan().as_hex() +
    "|" +
    manifest.layout_digest().as_hex() +
    "|" +
    expected_layer_index.to_string() +
    "|" +
    bank_layout.digest().as_hex() +
    "|" +
    routed.requirements_digest().as_hex() +
    "|" +
    routed.source_digest().as_hex() +
    "|" +
    routed.recipe_digest().as_hex() +
    "|" +
    shared.source_digest().as_hex() +
    "|" +
    shared.recipe_digest().as_hex()
  {
    execution_identity: manifest.execution_identity(),
    manifest_layout_digest: manifest.layout_digest(),
    layer_index: expected_layer_index,
    expert_bank_layout_digest: bank_layout.digest(),
    routed_recipe_digest: routed.recipe_digest(),
    shared_recipe_digest: shared.recipe_digest(),
    digest: Glm53MoeExpertBankCandidatePlanDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(digest_text))),
    ),
  }
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::digest(
  self : Self,
) -> Glm53MoeExpertBankCandidatePlanDigest {
  self.digest
}

///|
pub fn Glm53MoeExpertBankCandidatePlanDigest::as_hex(self : Self) -> String {
  self.0
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::execution_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::manifest_layout_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.manifest_layout_digest
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::expert_bank_layout_digest(
  self : Self,
) -> @glm53_weights.Glm53MoeExpertBankLayoutDigest {
  self.expert_bank_layout_digest
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::routed_recipe_digest(
  self : Self,
) -> @glm_hybrid_cuda_aot.GlmHybridCudaRecipeDigest {
  self.routed_recipe_digest
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::shared_recipe_digest(
  self : Self,
) -> @glm_hybrid_cuda_aot.GlmHybridCudaRecipeDigest {
  self.shared_recipe_digest
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::materialization_authority(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::upload_authority(_self : Self) -> Bool {
  false
}

///|
pub fn Glm53MoeExpertBankCandidatePlan::launch_authority(_self : Self) -> Bool {
  false
}
