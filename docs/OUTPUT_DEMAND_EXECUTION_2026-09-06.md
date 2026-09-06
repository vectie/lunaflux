# Output-demand execution: eager integration

Implemented in `c388721`. This is a conservative functional dead-suffix pass,
not per-row GPU compaction and not effect-only CUDA Graph capture.

At artifact/executor preparation, the model plan computes the ordered prefix
through its last persistent-state operation. Keeping that complete prefix
preserves every dependency and the order of KV updates. Eager executors retain
a shorter immutable queue sharing the existing allocations and functions.
Split launches with repeated semantic IDs remain included. Unknown/reordered
launch mappings retain the full executor.

During the existing descriptor-writing pass, a host scalar counts producing
rows. Zero selects the effect-only eager queue by index. No GPU descriptor ABI,
row offsets, model identity, RNG counter or KV-write range is overloaded or
changed. Zero-output frames also avoid sampling-result readback. Any producing
row, including a decode row, keeps the full queue. FP8 and full-graph diagnostic
canaries retain their existing execution contract.

Captures retain their full queue: this implementation does not duplicate graph
storage without accounting for its cost. It also does not compact output rows
inside mixed batches. Those are remaining work, not completed optimizations.

## Validation

- Warning-denied native check passed; device-step tests 152/152, model-plan
  tests 42/42. The optional GPU test is inactive in ordinary local runs.
- `scripts/validate-device-step-allocations.sh` passed: the existing allocation
  probe and single descriptor-scan assertions still hold.
- Exact committed source was uploaded and its optional physical white-box
  test was executed on the RTX 5060 Ti. The real ordered-executor preparation,
  enqueue/completion/reset and reverse cleanup paths were exercised.
- Alternating full/prefix/full/prefix execution advanced the retained effect
  counter four times and the head/sampling counters only twice. A subsequent
  captured full graph advanced all three, with no extra captured graph owner.
- Memcheck, racecheck, synccheck and initcheck all passed. Cleanup checks passed
  and no GPU compute process remained after the campaign.
- The Linux build emitted an existing third-party async C warning about
  `posix_spawn_file_actions_addchdir_np`; this was not empty-stderr validation.

This fixture validates queue semantics and shared-resource ownership, **not
full Qwen numerical equivalence or an end-to-end throughput improvement**.
Those measurements remain required before claiming a serving speedup.

Source archive SHA-256:
`60d536cb465c2b5264a734e4d9c8c260a62d46a6118351fe982f4c28968b229e`.
Results: `/private/tmp/lunaflux-output-demand-physical-20260906-r1.tar.gz`,
SHA-256 `679647e34cd40b4976a2e5fc3565f982a4bf819d2bcaa4ec2942cf098ee34e73`.
The archive includes the source archive, offline cubin, automation and all logs.
