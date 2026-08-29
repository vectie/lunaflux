///|
test "unsafe or ambiguous artifact locators fail closed" {
  for
    locator in [
      "", "/absolute", "../escape", "a/../b", "a//b", "a/./b", "a\\b", "é",
    ] {
    try
      @evidence.ReleaseEvidenceArtifact::new(
        locator,
        byte_length=1L,
        digest=@evidence.ReleaseArtifactDigest::from_sha256(
          "7777777777777777777777777777777777777777777777777777777777777777",
        ),
        limits=release_test_limits(),
      )
    catch {
      @evidence.InvalidArtifact => ()
      _ => fail("unexpected locator failure")
    } noraise {
      _ => fail("unsafe locator was admitted")
    }
  }
}

///|
test "status and artifact combinations cannot invent terminal evidence" {
  let artifact = release_test_artifact("evidence/a.bin", b"a")
  for status in [@evidence.Passed, Failed] {
    try
      @evidence.ReleaseEvidenceEntry::new(
        role=ReferenceCorrectness,
        status~,
        artifact=None,
        workload_count=0,
        losing_workload_count=0,
      )
    catch {
      InvalidEvidenceEntry => ()
      _ => fail("unexpected terminal evidence failure")
    } noraise {
      _ => fail("terminal evidence omitted its artifact")
    }
  }
  try
    @evidence.ReleaseEvidenceEntry::new(
      role=ReferenceCorrectness,
      status=NotRun,
      artifact=Some(artifact),
      workload_count=0,
      losing_workload_count=0,
    )
  catch {
    InvalidEvidenceEntry => ()
    _ => fail("unexpected NotRun evidence failure")
  } noraise {
    _ => fail("NotRun carried fabricated terminal evidence")
  }
}

///|
test "performance counts cannot hide or invent losing workloads" {
  let artifact = release_test_artifact("evidence/performance.bin", b"p")
  for values in [(0, 0), (4, -1), (4, 5)] {
    let (count, losses) = values
    try
      @evidence.ReleaseEvidenceEntry::new(
        role=Performance,
        status=Passed,
        artifact=Some(artifact),
        workload_count=count,
        losing_workload_count=losses,
      )
    catch {
      InvalidEvidenceEntry => ()
      _ => fail("unexpected performance count failure")
    } noraise {
      _ => fail("invalid performance counts were admitted")
    }
  }
}

///|
test "reviewer terminal state requires identity and attestation" {
  let identity = release_test_reviewer_identity(SecurityReviewer)
  let artifact = release_test_artifact("review/security.bin", b"r")
  for values in [(None, Some(artifact)), (Some(identity), None), (None, None)] {
    let (identity_value, attestation) = values
    try
      @evidence.ReleaseReviewerEntry::new(
        role=SecurityReviewer,
        status=Passed,
        identity=identity_value,
        attestation~,
      )
    catch {
      InvalidReviewerEntry => ()
      _ => fail("unexpected reviewer evidence failure")
    } noraise {
      _ => fail("terminal reviewer omitted identity or attestation")
    }
  }
}

///|
test "canonical admission rejects digest mismatch truncation and trailing bytes" {
  let record = release_test_record(Passed)
  try
    @evidence.admit_release_campaign(
      record.canonical_bytes(),
      expected_digest=@evidence.ReleaseCampaignDigest::from_sha256(
        "8888888888888888888888888888888888888888888888888888888888888888",
      ),
      limits=release_test_limits(),
    )
  catch {
    @evidence.CampaignDigestMismatch => ()
    _ => fail("unexpected digest mismatch failure")
  } noraise {
    _ => fail("wrong campaign digest was admitted")
  }
  let trailing_array = record.canonical_bytes().to_array()
  trailing_array.push(b'!')
  for
    changed in [
      record.canonical_bytes()[:record.canonical_bytes().length() - 1].to_owned(),
      Bytes::from_array(trailing_array),
    ] {
    let digest = @evidence.ReleaseCampaignDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(changed)),
    )
    try
      @evidence.admit_release_campaign(
        changed,
        expected_digest=digest,
        limits=release_test_limits(),
      )
    catch {
      @evidence.InvalidCanonicalRecord => ()
      _ => fail("unexpected malformed canonical failure")
    } noraise {
      _ => fail("malformed canonical record was admitted")
    }
  }
}

///|
test "canonical admission rejects a digest-valid foreign schema version" {
  let record = release_test_record(Passed)
  let changed = record.canonical_bytes().to_array()
  let version_offset = 8 + "lunaflux.release-campaign.v1".length()
  changed[version_offset] = b'\x02'
  let foreign = Bytes::from_array(changed)
  let digest = @evidence.ReleaseCampaignDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(foreign)),
  )
  try
    @evidence.admit_release_campaign(
      foreign,
      expected_digest=digest,
      limits=release_test_limits(),
    )
  catch {
    @evidence.InvalidCanonicalRecord => ()
    _ => fail("unexpected foreign-version failure")
  } noraise {
    _ => fail("foreign release-campaign version was admitted")
  }
}

///|
test "all-zero digest sentinels are rejected" {
  try
    @evidence.ReleaseTargetDigest::from_sha256(
      "0000000000000000000000000000000000000000000000000000000000000000",
    )
  catch {
    InvalidDigest => ()
    _ => fail("unexpected zero-digest failure")
  } noraise {
    _ => fail("zero digest sentinel was admitted")
  }
}

///|
test "record construction rejects reordered or incomplete obligations" {
  let entries = release_test_entries(Passed)
  let swapped = entries[0]
  entries[0] = entries[1]
  entries[1] = swapped
  try
    @evidence.ReleaseCampaignRecord::new(
      campaign_id=7UL,
      subject=release_test_subject(),
      image=release_test_image(),
      physical_target=release_test_target(),
      evidence=entries,
      reviewers=release_test_reviewers(Passed),
      limits=release_test_limits(),
    )
  catch {
    @evidence.InvalidEvidenceEntry => ()
    _ => fail("unexpected reordered-role failure")
  } noraise {
    _ => fail("reordered role set was admitted")
  }
}

///|
test "independent evidence roles cannot reuse one artifact locator" {
  let entries = release_test_entries(Passed)
  entries[12] = @evidence.ReleaseEvidenceEntry::new(
    role=ExternalAdapterBuild,
    status=Passed,
    artifact=entries[11].artifact(),
    workload_count=0,
    losing_workload_count=0,
  )
  try
    @evidence.ReleaseCampaignRecord::new(
      campaign_id=7UL,
      subject=release_test_subject(),
      image=release_test_image(),
      physical_target=release_test_target(),
      evidence=entries,
      reviewers=release_test_reviewers(Passed),
      limits=release_test_limits(),
    )
  catch {
    @evidence.InvalidArtifact => ()
    _ => fail("unexpected duplicate artifact failure")
  } noraise {
    _ => fail("independent roles collapsed onto one artifact")
  }
}

///|
test "independent evidence roles cannot reuse content under another locator" {
  let entries = release_test_entries(Passed)
  guard entries[11].artifact() is Some(runtime_build) else {
    fail("missing runtime build artifact")
  }
  let copied = @evidence.ReleaseEvidenceArtifact::new(
    "evidence/external-copy.bin",
    byte_length=runtime_build.byte_length(),
    digest=runtime_build.digest(),
    limits=release_test_limits(),
  )
  entries[12] = @evidence.ReleaseEvidenceEntry::new(
    role=ExternalAdapterBuild,
    status=Passed,
    artifact=Some(copied),
    workload_count=0,
    losing_workload_count=0,
  )
  try
    @evidence.ReleaseCampaignRecord::new(
      campaign_id=7UL,
      subject=release_test_subject(),
      image=release_test_image(),
      physical_target=release_test_target(),
      evidence=entries,
      reviewers=release_test_reviewers(Passed),
      limits=release_test_limits(),
    )
  catch {
    @evidence.InvalidArtifact => ()
    _ => fail("unexpected duplicate content failure")
  } noraise {
    _ => fail("independent roles reused one artifact under two locators")
  }
}
