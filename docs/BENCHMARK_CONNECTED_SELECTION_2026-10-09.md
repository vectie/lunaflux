# Connected compiler selection: implementation and serving test

## Result

**The new binding works; an additional performance advantage is not proved.**
The actual old selector, supplied identical complete-plan medians, sample counts
and resource constraints, chooses the same three plans as the new integration.
The existing best bundle already contains the long-input winners.

Implementation commit: `7e65a25f`. Keep the functional IR and consistent artifact
binding; do not credit the compiler architecture with a speedup when it selects
the same executable. This closes a specific offline selection/export gap, not
the broader task of generating better producer/consumer layouts.

## What is implemented

- `compiler/fusion_regions/connected.mbt`: immutable executable-region covers,
  matching live-boundary contracts, complete-chain measured selection through
  the existing frontier, and retained implementation vectors.
- Unknown consumer local-memory requirements are explicit, not zero. A finite
  local-memory constraint excludes unknown requirements. This experiment omits
  that constraint for both selectors and relies on existing artifact/runtime
  admission. Known reusable projection workspace remains bounded at 32 MiB.
- The real bundle exporter accepts `--prefill-chain-comparison PATH SHA256`.
  It binds both attention slots, projection policy and the chosen module's
  digest-checked decode-route table together. Fixed export context, cubin
  identities and numerical permission are checked offline.
- No request-path tuning, new kernel, new IR layer or serving-default change.
  All runs use the same frozen worker. The generic pass remains pure MoonBit;
  file loading/export and CUDA-specific catalog facts stay outside it.

The catalog here is four **complete serving configurations** sharing an existing
materialized BF16 boundary. It is not a new set of nonmaterialized consumer
layouts, automatic conversion generation, or whole-model layout search.

## Protocol and calibration

AKO's fixed-artifact, bounded-budget and separate-validation workflow was used.
Spark .179: GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
PCI `0000000F:01:00.0`. Qwen3-0.6B BF16, fixed synthetic varied token-ID prompts,
greedy generation, 64 output tokens/request. The prompt generator uses a small
12-token pool, not a natural-language quality corpus.

Four plans: KV128/KV64 reference attention crossed with generated/vendor
projection policy. The vendor policy applies at its existing 2,048-row boundary;
this is not a claim that every projection uses a vendor library.

Eight fresh calibration starts in order `0,1,3,2,2,3,1,0`; one warmup plus three
measured waves/cell/start. Each median below has six observations from two
independent starts, not six independent experiments. All alternatives retained:

| Input / output / concurrency | P0 KV128/generated ms | P1 KV128/vendor ms | P2 KV64/generated ms | P3 KV64/vendor ms | Frozen choice, old = new |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 / 64 / C8 | 714 | 708 | 711.5 | 709 | P1 |
| 8192 / 64 / C8 | 4841 | 4721 | 4749.5 | 4599.5 | P3 |
| 32512 / 64 / C1 | 3959 | 3916 | 3765 | 3700 | P3 |

**Do not interpret the cross-plan differences as isolated attention savings.**
The final route audit found a protocol limitation described below.

## Independent validation

Selections were exported and frozen before validation. The six starts were
control P3, automatically exported P1, automatically exported P3, P3, P1,
control P3. Every start ran all three cells. No reselection from validation.

| Input / output / concurrency | Current best control ms | Automatic selection ms | Selected output tok/s | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 512 / 64 / C8 | 708 | 708 | 723.16 | No gain; calibration's 1 ms lead disappeared |
| 8192 / 64 / C8 | 4594 | 4609 | 111.09 | Same bundle; +0.33% observed timing variation |
| 32512 / 64 / C1 | 3693 | 3700.5 | 17.29 | Same bundle; +0.20% observed timing variation |

Control versus selected median TTFT: 84/85 ms, 1343.5/1345.5 ms and
2232.5/2243 ms, respectively. Short-cell fresh-start completion medians were
707/713 ms for control versus 710/706 ms for selection. Thus even the direction
does not repeat reliably. The long-cell configurations are byte-identical;
their small timing differences are not a newly emitted execution change.

All validation arms, including P1's slower long cells, are retained in the JSON.
P1 validation medians there were 4727.5 ms at 8K and 3916.5 ms at 32K. These
were not used to change the frozen choices. No fresh vLLM/SGLang comparison,
unseen-shape test, second-device generalization or compiler-time benchmark was
performed in this follow-up.

## What the audit does and does not prove

The control bundle matches today's best bundle byte-for-byte:
`4255abca16b1edcbd3ae32c44c65eec550618e097a3ecdc2f6cc897d3861c77b`.
Each automatically exported bundle matches its corresponding manual candidate.
Validation launches use those automatically exported bundles. This proves
binding and propagation for the tested catalog, not just a selected-ID printout.

The unmodified old package is revision
`24ef8d5f5f45fe8f576bb006c1a3d2444adeeeb3`. Its diagnostic caller now accepts
`--samples 6`; both selectors receive the same six-observation medians. Local
memory is projected out for both, not falsely reported as a zero physical
requirement. Old/new whole-plan choices agree **3/3**. This is not a separate
old-runtime serving benchmark: both select the same existing artifacts.

The intended fixed decode policy was verified initially only for nominal C1/C8
rows. The broader final audit also checks intermediate buckets and found:

| Bucket | KV128 configuration | KV64 control |
| --- | ---: | ---: |
| decode / 4 rows / 4 queries / 16384 context bucket | candidate 1 | candidate 4 |

The actual selector requires a 1% improvement over its matched baseline; a
raw minimum is not its policy. Filtering authentic records while preserving
their baseline and scope therefore did not freeze every intermediate route.
This bucket can matter during the 8K/C8 case. No selected-route timeline was
captured here to quantify whether/how often it executed. The sealed result is
explicitly `completed-with-protocol-limitation`, not a clean isolated factorial
experiment. Complete-configuration timings and identical-artifact binding remain
valid; attention-only attribution and a demonstrated consumer-layout advantage
do not follow. The original failed strict audit is preserved.

Per-request files were compared by explicit row identity, not asynchronous
aggregate order. C1/32K vectors repeat and match across all four plans. C8
vectors vary even within the unchanged control. This repeats the limitation
already recorded in the [preceding report](BENCHMARK_REFERENCE_PREFILL_SCHEDULE_2026-10-09.md),
not a new correctness proof or a diagnosis of its cause. All requests returned
64 tokens; that alone is not model-quality or batch-invariance validation.

## Tests, resources and reproducibility

- Clean isolated Linux `moon info`, `moon fmt --check`, native check, full test
  and release exporter build passed: **3483/3483**. Existing repository warning
  exclusions remain `-79-20-29-25-92-14`; new fusion-region tests pass strictly
  without these exclusions, **16/16**. Exporter focused tests: **24/24**.
- All fourteen serving starts drained, closed their worker and had empty runtime
  stderr. GPU returned idle. Minimum sampled available host memory:
  **103667604 KiB = 98.87 GiB**, above the 32 GiB reserve. Serving cgroups were
  capped at 64 GiB with no swap and 900-second deadlines.
- The initial parallel CPU build hit its 8 GiB cgroup limit. Its failure is
  preserved; bounded two-job rebuilds passed, final peak 3.0 GiB. This was not
  a GPU run or host-memory exhaustion. Setup corrections for archive metadata,
  policy hash propagation, route admission and JSON integer encoding are also
  preserved; no failed measurement was replaced.
- Existing qualified cubins are unchanged. No new sanitizer or hardware-counter
  claim is made for this offline compiler/binding change.

Remote evidence:
`/home/wlc004s/lunaflux-connected-20261009.5TIzrCFV`.
Local verified evidence:
`/tmp/lunaflux-connected-20261009.QTZrMS/verified`.
All **3699** manifest entries verify. Archive SHA-256:
`06b09500f74aa5a6fbf590f66a009e3e5eb45b3af76f97acb9b6bc57b34df21c`.
Final local analysis: `/tmp/lunaflux-connected-20261009.QTZrMS/report-v2.json`,
SHA-256 `d853dcd28e5a264e67f997aaa0c0cd1d2a1d1b24d813917fbfef42a70ecfd7f0`.

Reproduce with `benchmarks/compiler_selection_ablation/connected*.mbtx` and the
shared old/new `setup.mbtx` / `selector.mbt`. Source is preserved as base
`865dfe08` plus the final overlay and the committed `7e65a25f` archive. No
production deployment or default promotion was performed.

## Decision

Keep the implementation as an opt-in offline binding capability, and keep the
current best serving default. Do not roll back the functional compiler on the
basis of this test, but do not claim the new frontier is a faster optimizer
than the old selector under equal information. To demonstrate additional
architectural performance value, the catalog must expose better executable
continuations/layouts, not merely add another selector around the same winners.
