# C2 numerical trajectory and dual-Spark decode measurements

## Outcome

This continues the measured-prefill dispatch experiment, not a fresh framework
comparison. Both Sparks are usable and performed separate work concurrently:
.179 reproduced real serving trajectories; .178 measured the existing decode
chains with runtime bucket geometry.

**The C2 differences are not a GPU sampler/readback disagreement.** All 3,072
observed GPU tokens agree with independent CPU argmax over the final BF16
logits. Small pre-token logit differences become exact ties, which greedy
sampling resolves by the lowest token ID; subsequent differing histories
amplify one first decision into many different output tokens.

**Disabling partitioned decode does not solve batch invariance.** Preserve
that failed causal intervention. Do not attribute the whole problem to the
partitioned decode kernel, change tie-breaking to fit the fixture, relax
tolerances, or call changed token vectors a demonstrated model-quality failure.
These results locate a numerical execution difference, not its unique first
producer or a proven memory race.

On .178, the existing eight-part decode chain is faster than ordinary decode
on all three tested long-history cells. This is a kernel-chain result, not a
new serving speedup; measurements from .178 must not become .179 route records.

## .179: real saved-input replay

Frozen model: Qwen3-0.6B BF16. The exact two saved input vectors each contain
32,512 input tokens and request 64 greedy output tokens. Model, AOT modules,
deployment policy and measured prefill table are unchanged. A disposable
worker adds synchronous final-logit readback and execution markers. It is
never used for performance or deployment.

Three fresh starts use scheduler chunk budgets 2048, 8192, 2048. Each start
runs solo prompt 0, solo prompt 1, pair 0/1 and reversed pair 1/0, twice:
12 requests / 768 output observations per start. A separate 2048 start disables
only the pure-decode partition-choice branch, retaining all other choices.

| Start | GPU/CPU argmax mismatches | Exact top-two ties |
| --- | ---: | ---: |
| Measured 2048, first | 0 / 768 | 35 |
| Measured 8192 | 0 / 768 | 32 |
| Measured 2048, second | 0 / 768 | 34 |
| Ordinary-decode intervention | 0 / 768 | 33 |

The normal starts' solo vectors and six tracked BF16 logits repeat exactly
within and across chunk budgets. Pair trajectories depend on which samples
execute solo, mixed or with two decode rows. Even the 8192 arm exhibits
solo/pair differences here; its previously stable four C2 repeats were a
finite capture, not a proof of general batch invariance.

The analysis reconciles client vectors with request-ID/generation margin
groups by their entire 64-token vector, rather than assuming request arrival
order. Ambiguity is retained explicitly if vectors are identical. All records
require complete unique sample indices; the final JSON retains every record.

### First differing decisions, before generated histories diverge

Representative first normal 2048 start:

| Prompt / execution | Sample (zero-based) | Best / second logits | Selected token |
| --- | ---: | --- | ---: |
| Prompt 0, solo | 4 | token 389 = 18.000; 315 = 17.875 | 389 |
| Prompt 0, paired | 4 | token 315 = 17.875; 389 = 17.875 | 315 |
| Prompt 1, solo | 2 | token 389 = 18.250; 13 = 18.125 | 389 |
| Prompt 1, reversed pair | 2 | token 13 = 18.250; 389 = 18.250 | 13 |

Prompt 0's paired sample 4 differs in 41/64 output positions after propagation;
the reversed-pair prompt 1 differs in 28/64. Those are not 41 or 28 independent
kernel errors. Both first decisions have identical preceding generated tokens.
Tracked logits can already differ before the first differing selected token.

### Executed routes explain why the prefill-only fix is incomplete

For paired prompt 0 in the first start:

| Sample | Actual prefill/decode rows | Actual token count | Bucket | Executed owner |
| --- | --- | ---: | ---: | ---: |
| 0 | 2 / 0 | 2048 | 540 | 27 |
| 1 | 1 / 1 | 1537 | 4068 | 70 |
| 2–4 | 0 / 2 | 2 | 2094 | 77 |

The sample-1 trace explicitly identifies the unmeasured mixed bucket selecting
split-prefill owner 69 (`kind=16`, family 2), then its mixed companion owner 70.
The new measured table covered **prefill**, not this **mixed** bucket. Source:
`engine/device_step/graph_bucket.mbt` considers both phases in the fallback.
This is an identified coverage gap, but not yet proof that it uniquely causes
the later sample-4 difference.

Pure solo decode normally selects owner 85; disabling split choice selects
owner 75. The two-row decode retains owner 77. The source fallback caps the
blockwise partition route at one row because prior measurements justified only
that case: `engine/device_step/paged_decode_split_prepare.mbt`. Changing that
cap is not equivalent to measuring or binding a per-device route record.

The intervention still yields solo/pair differences of up to 41 output
positions and changes some solo trajectories too. No production default is
changed. Next isolation must distinguish mixed-prefix KV/numerics from pure
decode and projection by observing surviving activation boundaries before
generated histories differ, with matching semantic inputs and selected chains.

## .178: paired ordinary versus partitioned decode

Same selected c468 module bytes as the frozen .179 serving exporter:
`b4e24a7bd1014106c55f0eb215a4dff2cfa7c29e4c33ee2edb60b97fc4a5dd7a`.
Recipe law: `owned8-blockwise-f32-probability-v4`; partial/merge use the module's
v3 symbols, not the separately emitted legacy v2 artifact. Both launches and
workspace are included. Geometry: ordinary X = rows, Y = 8, Z = 1;
partitioned X = rows, Y = 8, Z = 8, plus merge. No full-capacity overlaunch.

| Rows / history | Ordinary median ms | Partitioned median ms | Median-time reduction |
| --- | ---: | ---: | ---: |
| 1 / 32513 | 1.338 | 0.577 | 56.9% |
| 2 / 32513 | 1.337 | 1.161 | 13.2% |
| 2 / 32515 | 1.357 | 1.161 | 14.5% |

Five alternating-order pairs per cell, unchanged artifact hashes, empty stderr,
unchanged KV, bitwise equal complete outputs, and sampled independent FP64
error within 0.003. Registers/thread = 124; predicted resident blocks = 2;
local bytes = 0. These static resource values are not new Nsight stall counters.
Bitwise equality here does not imply arbitrary-input or whole-model parity.

The first .178 attempt failed before GPU launch because the emitted-source
directory did not contain its compiled cubin. That attempt is retained.
The corrected attempt copies the exact authenticated serving compilation,
verifies its hash, and runs under a new non-overwriting trial directory.

## Boundaries, memory and retention

GPU .179: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
GPU .178: `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
Both GB10/sm121. Pinned CUDA 13.0.88 compiler hash:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Preparation: 16 GiB/no swap/1200 seconds. .179 supervising replays: 48 GiB/no
swap, with the existing bounded serving/bridge units and 32 GiB MemAvailable
reserve. Minimum sampled available memory is 101,776,520 KiB for the three
normal starts and 101,200,564 KiB for the intervention. .178 probes: 8 GiB/no
swap/600 seconds; minimum before/after reserve is 122,212,380 KiB. Both GPUs
are idle after cleanup. No production kernel bytes, scheduler, numerical
contract or runtime policy were modified.

All three archives downloaded without overwrite to
`/tmp/lunaflux-c2-decode-transfer-20261005.1wZj4owK`; local archive hashes and
all manifest entries verify:

| Experiment / remote root suffix | Archive SHA-256 | Manifest entries |
| --- | --- | ---: |
| .179 `lunaflux-ako-chunk-trace-c2-20261005.cDXN4o4x` | `62c0401709e5f7a03ce0c213327c7c46b9c78eb2684ee9bc383a90a59cf31682` | 295 |
| .179 `lunaflux-ako-chunk-trace-c2-ordinary-20261005.wkkMpwgt` | `18793249926f0a467747533a4060060fc76bbbefb42a96e88fead58962731ae4` | 144 |
| .178 `lunaflux-ako-c2-decode-20261005.cle6wsdG` | `02c52cdd94d59065d4c9a3e176c9965a8d8594e133728749a33e68fc6bae61c5` | 57 |

Automation: bounded replay/intervention, identity-safe margin report, exact
decode contract, and an explicit ordinary-versus-partitioned probe option.
Focused warning-denied checks and four native script tests pass. This is not a
whole-repository phase boundary or a fresh vLLM/SGLang run. The prior measured
C1 gains remain the latest valid end-to-end performance result; no diagnostic
readback timing is counted as a new speedup.
