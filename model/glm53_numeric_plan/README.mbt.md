# GLM-5.3 numeric planning

This package binds the exact ordered GLM-5.3 logical tensor manifest and
hybrid execution schedule to backend-neutral `numeric_contract` storage and
operation requirements. The immutable binding covers profile, authenticated
model-plan identity, tensor-layout identity, numeric-schema identity, semantic
operation order, and logical tensor roles.

The admitted converted full profile and official Flash-BF16 profile retain
plain BF16 parameters plus their exact plain-F32 routing, KDA, and mHC control
tensors. The official full FP8 block-scale layout is deliberately rejected:
the generic numeric vocabulary has no 128-by-128 block-scale granularity, so a
per-tensor scale would be a false contract. Mixed-dtype physical
materialization is also rejected independently.

This is startup numeric planning only. It does not parse payloads, allocate or
materialize weights, select kernels, admit a device, or execute a model.
