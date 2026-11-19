# MiniCPM-V 4.7 startup preparation

This package joins the deployment-staged config, tokenizer, tensor index and
bounded shard headers into immutable startup plans. It owns no CUDA resources
and reads no tensor payload. The runtime owner consumes the resulting text
program and weight plan; the executable owns command parsing and result files.

The initial explicit Spark profile is one sequence, 32 rows and 4096 context
tokens. It retains the full 40-layer MiniCPM-V 4.7 35B-A3B decoder. Both source
export and resident startup call the same attention/program builders. The
declared `sm121` profile remains unmeasured and unqualified; it is not device
observation. Changing that profile requires rebuilding the matching AOT modules.

```mbt check
///|
fn _prepare_queue(
  root : @approved_fs.ApprovedRoot,
  document : Json,
) -> (
  @minicpm_v47_bootstrap.PreparedModel,
  ReadOnlyArray[@minicpm_v47_bootstrap.PreparedRequest],
) raise {
  let model = @minicpm_v47_bootstrap.prepare(root)
  let requests = @minicpm_v47_bootstrap.prepare_requests(
    model.model,
    model.tokenizer,
    document,
  )
  (model, requests)
}
```

`prepare` follows `tokenizer.json`, binds actual BF16 text tensor shapes and
joins every tensor-index owner to its inspected header. It does not hash model
or tokenizer payloads; legacy digest fields carry a supplied zero association
label. Deployment remains responsible for staging the pinned revision.
`read_asset` and `streaming_limits` bound metadata and startup transfer sizes.
Inspection requires the staged shard files; independently downloaded header
JSON is input to the separate source preparation command.

`prepare_requests` accepts a queue of up to 32 stateless system/user requests,
tokenizes them before device startup and preserves the prepared token IDs and
positions. The queue explicitly selects greedy generation; it does not reproduce
the upstream stochastic defaults. Unsupported media/sampling fields, duplicate
IDs, fractional/excessive token limits and prompt overflow are rejected. No
tools, images, multi-turn history or streaming API is implied by this profile.

The real-asset tests use the full actual config/tokenizer and raw tensor-header
fixtures with sparse zero payload holes. They do not provide checkpoint weights
or evidence of GPU logits. Set `LUNA_MINICPM_V47_ASSET_DIR` to the resolved pinned
assets when running those tests.
