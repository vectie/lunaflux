# Selected-token numerical diagnosis

## What was measured

Diagnostic-only worker builds read back the existing BF16 final logits after
GPU greedy sampling and independently compute the top two values on the CPU.
Request generation, sample index and logical position accompany each record.
The readback and markers are installed only in a disposable source copy;
these runs are **not performance measurements**.

| Pair | Outputs per variant | GPU/CPU argmax mismatches | Exact top-two ties |
| --- | ---: | ---: | ---: |
| c322, ragged input and staggered workloads | 12,288 | 0 | 83 |
| c324, same request bodies | 12,288 | 0 | 75 |
| Partial ingress, uniform-distinct 3072/32 | 3,072 | 0 | 4 |
| Full ingress, same request bodies | 3,072 | 0 | 5 |

Each pair includes C8/C16, a warmup and three repeated trials. Counts in this
report include warmups. Attention comparison covers 192 requests per variant;
the ingress comparison covers 96. Full ingress was confirmed by exporter
`ingress_schedule=full`; the partial bundle reports `producer-separated`.

## First divergence, not downstream token cascades

The attention pair differs on 45/192 requests. Every matched first-divergence
top-two margin is 0 or 0.125. Examples include:

- Sample 22: c322 selects 19 with logits 17 versus 16.875 for token 16;
  c324 ties tokens 16 and 19 at 17 and correctly selects the lower token ID.
- Sample 4: tokens 576 and 13583 exchange rank at a margin of 0 or 0.125.
- Samples 10, 32 and 51 likewise exchange close-ranked tokens.

The full/partial pair differs on 4/96 requests, all at sample 31, between
tokens 16 and 22. Matched margins are again 0 or 0.125. This is a new controlled
uniform-distinct corpus, not a rerun of the earlier 44-divergence corpus.

The comparator verifies identical request bodies and matches trace groups by
the complete output token vector and initial logical position. It retains
**all** matching groups when repeated requests have identical outputs; it does
not pretend that request IDs across separate processes identify the same
client. All divergent responses have matching trace groups. Some matches have
more than one margin, and the conclusions above include every matching margin.

## Conclusion and limit

For these 30,720 observed outputs, GPU sampling implements the CPU argmax of
the exact same logits. The divergences are upstream numerical ranking changes,
not an observed greedy-selection bug. Close BF16 ranks explain sensitivity of
the token decision, but **do not prove the upstream arithmetic is correct** or
establish an acceptable model-quality tolerance. A first-differing activation
comparison or independent higher-precision model reference is still required
before accepting these changes under a token-agreement contract.

The c322/partial baseline remains selected. Neither a small microbenchmark win
nor this margin diagnosis promotes c324 or full ingress. No production hot-path
readback, diagnostic logging, or new tolerance was introduced.

## Reproduction and artifacts

Tools in `benchmarks/gpu_pipeline`:

- `install_logit_margin_trace.mbtx`: applied after the current execution trace
  installer to a disposable source tree; asserts the diagnostic stderr seam.
- `summarize_logit_margins.mbtx`: validates mapped row identities and emits
  logits, margins and sampler agreement.
- `compare_logit_margins.mbtx`: compares response bodies/tokens and retains
  ambiguous matching trace groups explicitly.

Remote successful roots: `/tmp/lfmargins-20260914-r3` and
`/tmp/lfmarginfull-20260914-r2`. Earlier failed setup/client attempts remain
excluded, including the first 3072-token client retaining a 2048-token count
assertion; the corrected client ran in a fresh output directory.

Downloaded archive `/tmp/lunaflux-logit-margin-diagnostics-20260914.tar.gz`,
verified local and remote SHA-256:
`e4ec0c8c4ef048c6652eadd529c5565c9267118b22d7d519b6e9a6c732ab8279`.
It contains raw diagnostic traces, request/response bodies, summaries and
comparison outputs; deployment/model trees and build caches are excluded.
