///|
struct FusedCampaignObservation {
  qkv_absolute_error_ppb : Int64
  qkv_relative_error_ppb : Int64
  qkv_dispatch_canary : UInt
  residual_absolute_error_ppb : Int64
  residual_relative_error_ppb : Int64
  residual_dispatch_canary : UInt
}

///|
fn campaign_field(token : StringView, key : String) -> String raise {
  let prefix = key + "="
  require_fused_probe(
    token.has_prefix(prefix) && token.length() > prefix.length(),
    "physical observation field mismatch: " + key,
  )
  token[prefix.length():].to_owned()
}

///|
fn campaign_int(token : StringView, key : String) -> Int raise {
  let field = campaign_field(token, key)
  let value = @string.parse_int(field, base=10) catch {
    _ => raise FusedCudaProbeRejected("invalid physical integer: " + key)
  }
  require_fused_probe(
    value.to_string() == field,
    "non-canonical physical integer: " + key,
  )
  value
}

///|
fn campaign_int64(token : StringView, key : String) -> Int64 raise {
  let field = campaign_field(token, key)
  let value = @string.parse_int64(field, base=10) catch {
    _ => raise FusedCudaProbeRejected("invalid physical int64: " + key)
  }
  require_fused_probe(
    value.to_string() == field,
    "non-canonical physical int64: " + key,
  )
  value
}

///|
fn campaign_uint(token : StringView, key : String) -> UInt raise {
  let field = campaign_field(token, key)
  let value = @string.parse_uint(field, base=10) catch {
    _ => raise FusedCudaProbeRejected("invalid physical uint: " + key)
  }
  require_fused_probe(
    value.to_string() == field,
    "non-canonical physical uint: " + key,
  )
  value
}

///|
fn parse_fused_campaign_observation(
  bytes : Bytes,
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
) -> FusedCampaignObservation raise {
  let document = @utf8.decode(bytes) catch {
    _ => raise FusedCudaProbeRejected("physical observation is not UTF-8")
  }
  let lines = document.split("\n").to_array()
  require_fused_probe(
    lines.length() == 2 && lines[1] == "",
    "physical observation must be exactly one newline-terminated line",
  )
  let fields = lines[0].split(" ").to_array()
  require_fused_probe(
    fields.length() == 20,
    "physical observation field count drifted",
  )
  require_fused_probe(
    campaign_field(fields[0], "outcome") ==
    "fused-parallel-sm120-qualification-pass",
    "physical observation did not pass",
  )
  let cycles = campaign_int(fields[1], "cycles")
  require_fused_probe(cycles >= 1 && cycles <= 32, "invalid observed cycles")
  require_fused_probe(
    campaign_int(fields[2], "shapes") == 4 &&
    campaign_int(fields[3], "launches") == cycles * 20,
    "physical shape or launch count mismatch",
  )
  let qkv_absolute_error_ppb = campaign_int64(
    fields[4],
    "qkv_max_abs_error_ppb",
  )
  let qkv_relative_error_ppb = campaign_int64(
    fields[5],
    "qkv_max_relative_error_ppb",
  )
  let qkv_dispatch_canary = campaign_uint(
    fields[6],
    "qkv_observed_dispatch_canary",
  )
  let residual_absolute_error_ppb = campaign_int64(
    fields[7],
    "residual_max_abs_error_ppb",
  )
  let residual_relative_error_ppb = campaign_int64(
    fields[8],
    "residual_max_relative_error_ppb",
  )
  let residual_dispatch_canary = campaign_uint(
    fields[9],
    "residual_observed_dispatch_canary",
  )
  require_fused_probe(
    campaign_field(fields[10], "standalone_qkv_oracle") == "pass" &&
    campaign_field(fields[11], "deterministic_cubins") == "2x5" &&
    campaign_field(fields[12], "target") == "sm_120" &&
    campaign_field(fields[13], "device_uuid").has_prefix("GPU-") &&
    campaign_field(fields[14], "device_pci").length() == 16 &&
    campaign_field(fields[15], "device_name_sha256").length() == 64 &&
    campaign_int64(fields[16], "device_total_memory_bytes") > 0L &&
    campaign_int(fields[17], "cuda_driver_version") > 0 &&
    campaign_field(fields[18], "resources") ==
    "context0,stream0,event0,graph0,graph_exec0,allocation0,module0,function0,device_bytes0,pending0,cleanup0" &&
    campaign_field(fields[19], "authority") == "qualification-only",
    "physical identity, resource, or authority fact drifted",
  )
  let qkv = set.qkv_candidate()
  let residual = set.residual_norm_candidate()
  require_fused_probe(
    qkv_absolute_error_ppb >= 0L &&
    qkv_absolute_error_ppb <=
    qkv.numerical_policy().maximum_absolute_error_ppb() &&
    qkv_relative_error_ppb >= 0L &&
    qkv_relative_error_ppb <=
    qkv.numerical_policy().maximum_relative_error_ppb() &&
    residual_absolute_error_ppb >= 0L &&
    residual_absolute_error_ppb <=
    residual.numerical_policy().maximum_absolute_error_ppb() &&
    residual_relative_error_ppb >= 0L &&
    residual_relative_error_ppb <=
    residual.numerical_policy().maximum_relative_error_ppb(),
    "physical numerical observation exceeds fixed policy",
  )
  require_fused_probe(
    qkv_dispatch_canary == qkv.dispatch_canary() * 12U &&
    residual_dispatch_canary == residual.dispatch_canary() * 12U,
    "physical aggregate dispatch canary mismatch",
  )
  {
    qkv_absolute_error_ppb,
    qkv_relative_error_ppb,
    qkv_dispatch_canary,
    residual_absolute_error_ppb,
    residual_relative_error_ppb,
    residual_dispatch_canary,
  }
}

///|
fn require_clean_sanitizer_log(bytes : Bytes, tool : String) -> Unit raise {
  let document = @utf8.decode(bytes) catch {
    _ => raise FusedCudaProbeRejected(tool + " log is not UTF-8")
  }
  let mut summaries = 0
  for line in document.split("\n") {
    if line.contains("ERROR SUMMARY:") {
      require_fused_probe(
        line == "========= ERROR SUMMARY: 0 errors",
        tool + " contained a non-canonical error summary",
      )
      summaries += 1
    }
  }
  require_fused_probe(
    summaries == 1,
    tool + " did not contain exactly one clean error summary",
  )
}
