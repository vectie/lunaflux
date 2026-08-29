# Qwen3 BF16 v12 serving profiles

The Qwen3 v12 materializer publishes one of two Qwen-only deployment profiles.
The default `native-framed-v1` profile accepts token-ID requests on the native
framed protocol. The explicit `openai-responses-v1` profile publishes an
instance-policy v3 document for authenticated OpenAI Responses SSE at
`/v1/responses`. They are separate launch bundles because one authenticated
instance policy selects exactly one external protocol.

The native correctness profile remains c1. The OpenAI profile is a benchmark
profile and therefore requires authenticated c32 release geometry: batch rows,
query rows, and query-token capacity must be present in the digest-pinned
release-bind receipt and must admit 32 rows. Only then does the materializer
derive the descriptor's plan/completion rows and the scheduler's active,
waiting, token-budget, and lane limits from that receipt. It never widens a c1
kernel release by editing policy JSON. The current c1 binder therefore exposes
the exact capacity blocker and the OpenAI materializer fails with `OpenAI
benchmark profile requires authenticated c32 release geometry` until a c32 AOT
release has been independently built and admitted.

The OpenAI profile uses the model alias `qwen3-0.6b-bf16` and renders messages
with the Qwen ChatML boundaries `<|im_start|>system`, `<|im_start|>user`, and
`<|im_start|>assistant`, their `<|im_end|>` suffixes, and a final assistant cue.
It retains the same dense-Qwen3 BF16 v12 model, numeric weights, AOT graph, KV
layout, and host sampling route as the native profile. It does not probe or
fall back to another model family.

## Materialization

Omitting the final profile argument is equivalent to `native-framed-v1`:

```text
scripts/materialize-qwen3-bf16-v12-launch.sh ... LISTEN_PORT NEW_OUTPUT native-framed-v1
scripts/materialize-qwen3-bf16-v12-launch.sh ... LISTEN_PORT NEW_OUTPUT openai-responses-v1
```

Materialize the two profiles into different new output directories. Each
invocation prints its exact launch SHA-256; the runtime argument is the output
directory suffixed with that digest.

## Opaque CLI descriptor contract

The production command is:

```text
lunaflux run ABSOLUTE_DEPLOYMENT_ROOT#sha256=LAUNCH_SHA256
```

The deployment supervisor must construct the inherited channels before exec.
They are capabilities, not ordinary files, argv values, or environment
variables.

- Fixed descriptor 5 is mandatory for both profiles and must be the process end
  of a connected Unix stream. To request deterministic drain, the supervisor
  writes the exact eight bytes `LFD1DRN\n`. LunaFlux replies with exactly one
  eight-byte terminal response: `LFD1ACK\n`, `LFD1IDM\n`, or `LFD1CLS\n`. The
  supervisor keeps its peer open until that response and process exit.
- Fixed descriptor 6 must be absent for `native-framed-v1`. It is mandatory for
  `openai-responses-v1` and must be the process end of a separate connected Unix
  stream. Before exec (or immediately after it), the supervisor writes one
  `LFC1KEY` frame: the eight bytes `LFC1KEY\n`, a little-endian unsigned
  credential length in bytes 8 and 9, two zero reserved bytes, and that many
  nonempty bearer-credential bytes. The OpenAI profile bounds the credential to
  128 bytes. The supervisor then shuts down the write side so LunaFlux observes
  EOF. Extra frames, a missing EOF, an empty credential, or a malformed length
  fail startup closed.
- Fixed descriptor 7 is absent for Qwen v12 because this recipe carries no Luna
  promotion approval.

OpenAI requests use `Authorization: Bearer CREDENTIAL` and the model alias
`qwen3-0.6b-bf16`. A serving or benchmark harness must wait for readiness,
exercise the SSE terminal event, request drain through fixed descriptor 5,
verify the drain response, wait for process/worker exit, and check that KV,
listener, child, CUDA allocation, and context ownership returned to zero.
