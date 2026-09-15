# Fixed selected-graph repetition

This diagnostic distinguishes fixed-plan repeatability from cross-batch
numerical identity. It does not measure inference performance or qualify the
current repository as a production release.

## Method and scope

An isolated historical trace worker executes the selected ordered graph,
waits for completion, snapshots producing rows' BF16 logits, resets the ordered
executor, and executes the same graph again without advancing the request.
It compares the resulting logits byte for byte. Positioned KV writes are
repeated in place: the original pre-step KV state is **not** restored. This
method assumes those writes are idempotent and is not valid for arbitrary
stateful graphs. A mismatch would still need investigation of that assumption.

The diagnostic adds allocation, synchronous readback and a second execution.
None of these operations is installed in the production tree or included in
reported performance. Kernel modules are the historical partial/full pair from
the preceding numerical investigation, not newly compiled current kernels.

## Initial result

On the RTX 5060 Ti, each runtime completed four trials at C8 and C16 with
3072 input / 32 output tokens and distinct deterministic token inputs.

| Historical runtime | Compared logits rows | Compared bytes | Changed rows |
| --- | ---: | ---: | ---: |
| Partial ingress | 3072 | 933494784 | 0 |
| Full ingress | 3072 | 933494784 | 0 |

No fixed-plan instability was observed in these comparisons. This does not
prove cross-batch equivalence, task-quality acceptance, or identify the first
operator causing the previously observed terminal-token differences.

The first attempt omitted `ordered.reset()` between completed executions and
failed before producing comparisons. It is a diagnostic state-machine error,
not a production-kernel failure; its logs remain preserved separately.

Review also found that producing rows need not form a batch prefix. The
installer now traverses the staged row count and selects rows through
`output_indices`, preserving the original row identity in each snapshot.
The table above predates that coverage correction.

## Corrected mixed-row rerun

The corrected worker repeats both uniform-distinct and ragged-staggered
requests at C8/C16, with two trials each. Ragged input lengths are
63/65/511/513/3583/3585/4031/4033; output lengths are
16/112/32/96/48/80/64/64, and arrivals are staggered by 25 ms per row.
This is a workload-coverage test, not an equal-work performance comparison.

| Historical runtime, corrected diagnostic | Compared logits rows | Compared bytes | Changed rows |
| --- | ---: | ---: | ---: |
| Partial ingress | 4608 | 1400242176 | 0 |
| Full ingress | 4608 | 1400242176 | 0 |

The compared row counts equal the requested output-token counts. Both runs
terminated successfully and the GPU compute-process query was empty afterward.
The result narrows the investigation to differences across execution contexts;
it does not establish that every other possible fixed graph is repeatable.

## Reproduction and remaining work

`benchmarks/gpu_pipeline/install_fixed_graph_repeat.mbtx` installs only into an
explicit disposable trace source. `summarize_fixed_graph_repeat.mbtx` refuses
empty comparison logs. `test_margin_analysis.mbtx` checks equal/different
records, empty-log rejection, reset placement, row selection and duplicate
installation rejection. The corrected worker also builds natively on Linux.

Downloaded initial results and the exact pre-row-fix diagnostic worker/source:
`/tmp/lunaflux-fixed-repeat-r2-results.tar.gz`, SHA-256
`4f63bcc8d19faded1f5a37db69928e18dbce1334ef21e92d661c42683b555ed5`.
Local and remote hashes agree. The corrected mixed-row archive is
`/tmp/lunaflux-fixed-repeat-r3-results.tar.gz`, SHA-256
`a78efb316ce8bb97f4d1acc2937d38e4bd40634a46d268ee3e644d4079cba6f6`,
also verified locally. It includes the corrected diagnostic worker/source,
client workload and logs, without model copies or credential-bearing launch
arguments.

Local validation: diagnostic regressions pass; `moon info`, format check,
warning-denied native check and the 3800/3800 full native suite pass in the
current working tree. Unrelated existing changes are not included in this
diagnostic commit, so that suite count is not a clean-commit release claim.

Still open: actual-model activation cut points under controlled batch/schedule
changes, remaining common strategy/effect
integration, and a fresh uninstrumented current-source serving benchmark.
The five compiler closeout workstreams are not declared complete by this test.
