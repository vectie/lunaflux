///|
fn authority_model_spec() -> LlamaModelSpec raise {
  LlamaModelSpec::new(
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
}

///|
fn authority_content_digest() -> ContentDigest raise {
  ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
test "LlamaModelSpec defensive validation rejects forged geometry in precedence order" {
  let valid = authority_model_spec()
  for
    case in [
      (
        { ..valid, vocabulary_size: 0, head_dimension: 7 },
        "InvalidDimension(Vocabulary)",
      ),
      (
        { ..valid, head_dimension: 7 },
        "InvalidHeadRelationship(HeadDimensionMismatch)",
      ),
    ] {
    let (forged, expected) = case
    try forged.validate() catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged model geometry rejection")
    }
  }
}

///|
test "LlamaModelSpec constructor rejects unsafe divisors before deriving head dimension" {
  for
    case in [
      (-2147483647 - 1, -1, "InvalidDimension(Hidden)"),
      (8, 0, "InvalidDimension(AttentionHeadCount)"),
      (8, -1, "InvalidDimension(AttentionHeadCount)"),
    ] {
    let (hidden_size, attention_head_count, expected) = case
    try
      LlamaModelSpec::new(
        vocabulary_size=32,
        hidden_size~,
        intermediate_size=16,
        layer_count=1,
        attention_head_count~,
        kv_head_count=1,
        context_length=16,
        rope_theta=10000.0,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected unsafe head divisor rejection")
    }
  }
}

///|
test "LlamaModelMetadata defensive validation rejects forged content" {
  let spec = authority_model_spec()
  let content = authority_content_digest()
  let valid = LlamaModelMetadata::from_verified_content(spec, content)
  let forged_digest : ContentDigest = { hex: "not-a-digest" }
  let forged = { ..valid, content_digest: forged_digest }
  try forged.validate() catch {
    error => debug_inspect(error, content="InvalidDigest")
  } noraise {
    _ => fail("expected metadata content rejection")
  }
}

///|
test "LlamaModelMetadata constructor revalidates a forged specification first" {
  let forged = { ..authority_model_spec(), head_dimension: 7 }
  try
    LlamaModelMetadata::from_verified_content(
      forged,
      authority_content_digest(),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="InvalidHeadRelationship(HeadDimensionMismatch)",
      )
  } noraise {
    _ => fail("expected metadata specification revalidation")
  }
}

///|
test "LlamaModelMetadata rejects forged content digest before retention" {
  let forged_content : ContentDigest = { hex: "not-a-digest" }
  try
    LlamaModelMetadata::from_verified_content(
      authority_model_spec(),
      forged_content,
    )
  catch {
    error => debug_inspect(error, content="InvalidDigest")
  } noraise {
    _ => fail("expected forged content digest rejection")
  }
}

///|
test "ModelIdentity constructor defensively validates digest components" {
  let forged_content : ContentDigest = { hex: "not-a-digest" }
  let forged_plan : PlanDigest = { hex: "also-not-a-digest" }
  try ModelIdentity::new(forged_content, forged_plan) catch {
    error => debug_inspect(error, content="InvalidDigest")
  } noraise {
    _ => fail("expected content digest precedence")
  }
  try ModelIdentity::new(authority_content_digest(), forged_plan) catch {
    error => debug_inspect(error, content="InvalidDigest")
  } noraise {
    _ => fail("expected plan digest validation")
  }
}
