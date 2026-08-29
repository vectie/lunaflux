///|
async test "plan command admits a real config and derives deterministic KV capacity" {
  let root = @fs.realpath("tests/reference_corpus")
  inspect(
    @lunaflux_plan.render(
      root, "config.json", "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622",
      "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30", "1048576",
      "16",
    ),
    content=(
      #|LunaFlux admitted startup plan
      #|model config sha256: 9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622; evidence: authenticated snapshot
      #|model content sha256: 852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30; evidence: caller-declared, not weight-verified
      #|model plan sha256: 22103eddad1277154b79321e53893ad579e47d040d62b896f5596624e0b7be3e; evidence: derived from authenticated config
      #|architecture: dense Llama; reason: authenticated bounded config
      #|execution semantics: paged KV
      #|operation count: 21
      #|maximum batch rows: 1
      #|maximum sequence tokens: 256
      #|vocabulary size: 3000
      #|hidden size: 16
      #|intermediate size: 64
      #|attention heads: 4
      #|KV heads: 4
      #|head dimension: 4
      #|semantic workspace ceiling bytes: 65536
      #|required kernel capabilities: 8
      #|  capability[0]: embedding_lookup (id 1); reason: required by admitted model plan
      #|  capability[1]: rms_norm (id 2); reason: required by admitted model plan
      #|  capability[2]: qkv_projection (id 3); reason: required by admitted model plan
      #|  capability[3]: positioned_rotary_embedding (id 9); reason: required by admitted model plan
      #|  capability[4]: paged_causal_attention (id 10); reason: required by admitted model plan
      #|  capability[5]: dense_projection (id 6); reason: required by admitted model plan
      #|  capability[6]: residual_add (id 7); reason: required by admitted model plan
      #|  capability[7]: gated_mlp (id 8); reason: required by admitted model plan
      #|KV budget bytes: 1048576
      #|KV physical page bytes: 2048
      #|KV tokens per page: 16
      #|KV physical page count: 512
      #|KV planned arena bytes: 1048576
      #|KV token capacity: 8192
      #|full-context sequence capacity: 32
      #|KV decision: maximum whole-page capacity within explicit budget; unused tail bytes remain outside the planned arena
      #|kernel catalog: unresolved; required capabilities are known, but no digest-pinned execution manifest was admitted
      #|model materialization: not performed
      #|device allocation: not performed
      #|readiness: false; kernel and physical-device gates remain open
      #|
    ),
  )
}

///|
test "plan command rejects invalid capacity before opening its root" {
  for invalid in ["0", "1099511627777", "not-a-number"] {
    try
      @lunaflux_plan.run(
        "/this/root/must/not/be-opened", "config.json", "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622",
        "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30", invalid,
        "16",
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid KV budget was admitted")
    }
  }
  for invalid in ["0", "4097", "not-a-number"] {
    try
      @lunaflux_plan.run(
        "/this/root/must/not/be-opened", "config.json", "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622",
        "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30", "1048576",
        invalid,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid tokens-per-page value was admitted")
    }
  }
}
