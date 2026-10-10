# GLM-5.3 Flash: real two-Spark no-hashing smoke

## Resident original-text sequence — 2026-10-11

Exact-source `121d04239d6fe7a2aacf9b7c3c53f0e94c5aa00b` now processes
two distinct original-tokenizer literal prompts (13 and 20 tokens) through one
resident pair of ranks. Both prompts were prepared and capacity-checked before
CUDA; the same row-one/history-64 AOT and model weights remain resident until
both requests retire. No TLS or payload hashing was introduced.

| Request | Input / output tokens | Generation lifetime, including prefill | Retirement epoch | Output IDs |
| --- | --- | ---: | ---: | --- |
| Original chat diagnostic | 13 / 8 | 429,489 ms | 21 | `785,1196,1101,3208,330,64,1,1959` |
| Literal English corpus case | 20 / 8 | 614,957 ms | 48 | `6771,594,1490,1246,419,4278,13,34542` |

Both requests finish `Length` with stopping disabled. The first matches the
previous isolated native run below; the second has no independent real-weight
reference yet. These timers exclude initial weight loading but include scalar
kernel execution, prefill, host/control work and transport; they are not
optimized serving, isolated decode token/s, or matched framework benchmarks.
Journal stdout is buffered until exit; its displayed line timestamps must not
be used to infer individual kernel or first-token times.

Both units report `Result=success`, `ExecMainStatus=0`, `SubState=exited`;
both explicit context-close markers are present. Ingress/egress cgroup memory
peaks are 53,924,024,320 and 51,160,498,176 bytes, respectively, with swap peaks
zero. Post-run GPU compute-owner queries are empty. They used the same 96-GiB
limits and planned arenas as below. No other GPU model ran concurrently.

Local and remote capture root: `/tmp/lunaflux-glm-sequence-run-20261011-v1`.
Units: `lunaflux-glm-sequence-run-20261011-v1-ingress.service` and
`lunaflux-glm-sequence-run-20261011-v1-egress.service`.
The shared output decoder subsequently adds a CPU-only `decode-tokens` entry
for GLM, preserving original special tokens/raw bytes without weights or CUDA.
Its affected frontend/CLI matrix passes 10/10; this is not model-logit parity.

## Result and scope

The ARM release built from `41650b1b8173dcc2d5aeca93d69f4f1702f16f2e`
completed a checkpoint-derived 13-token chat-template prompt and eight greedy
output tokens on Spark .178/.179. Both systemd **user** units exited normally
with `Result=success`, `ExecMainStatus=0`, and `MainPID=0`. Both terminal GPU
process queries were empty. No checkpoint payload hashes were calculated:
both startup journals explicitly report `payload_hashing=false` for 121 shards.

This is the scalar, row-one checkpoint diagnostic path, **not** the optimized
batched serving benchmark and **not** independent upstream numerical parity.
It reuses existing AOT artifacts. The new tokenizer `ignore_merges` work was
not in this runtime binary; prompt frames came from the previous exact-ASCII
chat-template diagnostic. Arbitrary-text checkpoint tokenization remains a
separate integration requirement.

Output IDs:

```text
785,1196,1101,3208,330,64,1,1959
```

## Memory and execution

| Rank | Layer range | Planned arena bytes | Observed cgroup memory peak | Swap peak |
| --- | --- | ---: | ---: | ---: |
| .178 ingress | [0,25) | 102,064,091,128 | 48,137,277,440 | 0 |
| .179 egress | [25,45) | 91,516,526,804 | 49,414,156,288 | 0 |

Each process had `MemoryMax=96G`, `MemorySwapMax=0`, and
`RuntimeMaxSec=7200`; the planner retained a 2 GiB reserve. Cgroup peaks are
not a measurement of all device allocations and must not be substituted for
the planned arena requirement. No DSpark or other GPU campaign ran concurrently.

Second-resolution journal observations (CST):

- Egress started 21:28:15 and reported weights loaded at 21:29:49: 94 s.
- Ingress started 21:28:50; first input completion appeared at 21:30:52.
- Final prompt completion / first output ID appeared at 21:34:56.
- Eight outputs completed at 21:37:43 with `finish=Length`.
- First-output-to-final interval was approximately 167 s for seven additional
  output tokens (about 23.9 s/token). Cold start to completion was 533 s.

These coarse diagnostic times include CPU/control/activation transport and
device execution. They are not isolated kernel timings. The historical run
spent roughly twenty minutes hashing before load; it used another binary and
output length, so it is **not** a matched speedup experiment. Removing the
scan demonstrably removes that work, but no percentage speedup is claimed.

## Retained results

Local non-overwriting download:
`/tmp/lunaflux-glm-nohash-result-20261010.qdOiDAU0` contains token output,
13 completion frames, ingress/egress journals, terminal unit status, and empty
terminal GPU queries. Capture command stderr files are empty. No checksum scan
was performed during download.

- .178 unit: `lunaflux-glm-nohash-ingress-20261010-v1.service`
- .179 unit: `lunaflux-glm-nohash-egress-20261010-v1.service`
- .178 output: `/tmp/lunaflux-checkpoint-build-glm-nohash-20261010-v1/output-eight`
- .178 model: `/tmp/lunaflux-glm-real-20261010.6yIBQN35/GLM-5.3-Flash-NVFP4`
- .179 model: `/tmp/lunaflux-glm-real-20261010.ekEYBzqU/GLM-5.3-Flash-NVFP4`
- .178 AOT: `/tmp/lunaflux-glm-real-aot-20261010.qdZHUGIP`
- .179 AOT: `/tmp/lunaflux-glm-real-aot-20261010.uvS6YcCC`

Open requirements: actual checkpoint arbitrary-text tokenizer/chat-template
integration; independent reference correctness; bounded batched serving and
matched framework benchmarks; stage-level performance attribution. This smoke
does not complete the three-model execution goal.
