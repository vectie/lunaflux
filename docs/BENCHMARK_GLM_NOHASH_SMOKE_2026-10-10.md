# GLM-5.3 Flash: real two-Spark no-hashing smoke

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
