# LunaFlux benchmark contract

## Purpose

Benchmarks decide whether an optimization ships. They do not decorate release
notes. Every comparison records enough information to reproduce model,
hardware, software, workload, warm-up, and measurement conditions.

## Baselines

Pin exact revisions or container digests for:

- LunaFlux;
- vLLM;
- SGLang;
- CUDA driver and toolkit;
- model and tokenizer artifacts.

The same GPU power mode, clocks policy, visible devices, model precision,
context limit, and request corpus apply to all engines.

## Correctness before speed

For each model/kernel change:

1. compare loaded tensor metadata and checksums;
2. compare intermediate operator fixtures where practical;
3. compare logits against the named reference tolerance;
4. compare deterministic greedy tokens;
5. test boundary shapes, empty/short inputs, maximum supported context,
   cancellation, and exhausted capacity;
6. run native sanitizer and leak campaigns.

A faster result with an unexplained output difference fails.

## Workload profiles

At minimum:

- latency: 128 input, 128 output, concurrency 1;
- chat: mixed 128–4096 input, 64–512 output;
- long prefill: 16K or maximum supported input, short output;
- decode-heavy: short input, 2K output;
- prefix-rich: repeated system prefix with varied suffixes;
- prefix-cold: unique prompts with equivalent lengths;
- saturation: increasing concurrency through queue formation;
- churn: cancellations and disconnects during prefill and decode;
- mixed: deterministic distribution of all supported profiles.

Token lengths are measured after each engine's tokenizer and verified to match.
Report tokenizer-included and engine-only results separately.

## Metrics

- request throughput and generated tokens per second;
- TTFT p50, p95, p99;
- inter-token latency p50, p95, p99;
- end-to-end latency;
- queue and tokenization time;
- prefill and decode tokens per scheduler step;
- batch and token-budget utilization;
- GPU utilization and memory high-water mark;
- KV capacity, hit rate, eviction, and recomputation;
- CPU utilization, scheduler time, and allocations;
- model load, warm-up, graph-capture, and readiness time;
- failure, deadline, cancellation, and output-backpressure rates.

## Measurement protocol

- Record hardware topology and thermals.
- Warm each engine using an identical declared procedure.
- Run randomized engine order to reduce temporal bias.
- Repeat enough trials to publish dispersion, not one best run.
- Separate cold-start from steady-state results.
- Include rejected and timed-out requests in outcome accounting.
- Store raw events and the script/config digest with the summary.

## Phase gates

- Reference phase: correctness only; no throughput claim.
- Online phase: stable behavior and bounded resource use under saturation.
- Paged-KV phase: within ten percent of the best pinned baseline on the
  supported workload envelope.
- Prefix/overlap phase: parity or better on at least one prefix-rich profile
  without material regression on prefix-cold profiles.
- MoonTile replacement: a custom kernel ships only when it beats the selected
  vendor/reference kernel on its declared shape set and does not regress the
  end-to-end mixed workload.
- Release: publish all results, including profiles where LunaFlux loses.

Thresholds are hypotheses until measured on the target hardware. Updating a
threshold requires an architecture decision and preservation of historical
results.

