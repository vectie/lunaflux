# Checkpoint shard inventory

This startup-only parser consumes a bounded sha256sum-style list of relative
`.safetensors` names and SHA-256 identities. It does not read weight payloads,
execute shell expressions, or bypass `inspect_shards` authentication. Other
model-root files such as tokenizer/configuration are deliberately not entries.
