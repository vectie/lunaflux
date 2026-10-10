# Base-attached DSpark prediction execution

`DeepSeekPredictionExecution` owns the local embedding copy, complete prepared
predictor, result/RNG ports and two phase executors. Base egress lends complete
ordered target captures and its vocabulary matrix. These borrowed allocations
must remain live until this owner closes. The extra footprint is explicit and
must be added to the base stage placement before its first allocation.

Submit only after global base commit. Prompt chunks use Prime; committed decode
uses Predict. While active, a capture lease prevents base overwrite, activation
transfer or close. Poll retires the queue and releases that lease. Closing drains
first, then releases queues, predictor state, result buffers and embedding.
The shared `prediction_phase_execution` package owns no model semantics; its
warmed replay is allocation-free in the native device double.

This is prediction attachment, not speculative accept/reject/commit. The
diagnostic executable can emit drafts without using them as committed output.
Full checkpoint numerical validation and distributed partial-capture assembly
remain separate work.
