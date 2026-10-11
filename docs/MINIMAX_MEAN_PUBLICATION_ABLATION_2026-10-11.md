# MiniMax encoder mean-publication ablation — 2026-10-11

This tests a specific source-level difference, not a new acceptance tolerance.
ATen's mean reduction publishes the F32 square sum multiplied by an F32
reciprocal. The selected native row normalization divides the sum by 5,120.
These can differ by one F32 ULP for this non-power-of-two width.

The offline source probe changes only that publication operation. Compensated
square accumulation, ordered projection dots, original weights, literal input
IDs, BF16 normalization/product boundaries and all fifty layers remain fixed.
The 128-wide head normalization is unchanged. Four exact-row AOT modules run
serially on Spark `.179`; the independent reference is the installed original
encoder, already verified against all 204 reconstructed hidden tensors.

| Input | Rows | Selected global relative L2 | Reciprocal global relative L2 | Reciprocal worst final-row relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Caption | 3 | 0.02942084% | 0.02942084% | 0.955769% |
| Digits | 56 | 0.4532132% | 0.4532398% | 3.123820% |
| Long | 140 | 0.7959295% | 0.7959280% | 9.053265% |
| English | 20 | 0.1079348% | 0.1079348% | 3.598594% |

Caption and English final outputs are bitwise identical to the selected native
default. Digits retains 225,024/286,720 identical values and long retains
602,166/716,800. Their worst final-row errors are unchanged. The few global-norm
changes do not fix the remaining propagated encoder differences; this variant
is **not selected for production**. It must not be described as a correctness
fix merely because it resembles one library operation more closely.

All four executions finish successfully and release their own GPU ownership.
The original long-run collection reported an erroneous owner failure: its CPU
download phase overlapped the next digits unit after the long GPU had already
retired. That global snapshot was not evidence of a leaked long-run allocation.
Commit `10262d14` fixes the offline collector to check its exact user unit's
MainPID, execution status and retired state. It still checks GPU availability
before dispatch. Three collector regressions pass; no token-step check or
production hardening is added. The original failed collection receipt remains.

## Reproduction

`scripts/probe-minimax-norm-mean.mbtx` prepares the isolated source change.
`scripts/run-minimax-text-encoder.mbtx` executes the exact-row modules and
collects 51 hidden tensors per case. `scripts/compare-minimax-encoder-reference.mbtx`
reports global and individual-row differences.

- AOT roots: `/tmp/lunaflux-minimax-request-aot-reciprocal-mean-{long,digits,actual-input,english}-20261011-v1`.
- Outputs: `/tmp/lunaflux-minimax-text-encoder-reciprocal-mean-{long,digits,actual-input,english}-20261011-v1`.
- Original comparisons: `/tmp/lunaflux-minimax-mean-{long,digits,caption,english}-vs-original-20261011-v1.txt`.
- Selected-default comparisons: `/tmp/lunaflux-minimax-mean-{long,digits}-vs-selected-20261011-v1.txt`.

The larger media-quality request is a separate experiment, currently running
on `.178` at `/tmp/lunaflux-minimax-joint-quality128-20261011-v1`: 128×128,
five seconds, seed 17, literal caption `A red cat`, twenty evaluations from
twenty-one original F32 inclusive-zero sigma points, all fifty blocks and both
VAEs. Its output has not yet been inspected. Neither this mean ablation nor
the earlier 32×32 blurred smoke output establishes audiovisual quality.
