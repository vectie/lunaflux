# Offline resource policy

Pure selection of an immutable backend-supported execution policy. Target
adapters supply candidate consumer/accumulator/transfer geometry and hardware
budgets. The generic selector contains no CUDA, model, or architecture branch.

Observations match exact device, toolchain, shape/layout/numeric contract,
program revision and generated artifact identity. The caller is responsible for
forming those identities from the complete compilation input. Invalid geometry,
duplicate identities and bounded-table overflow are rejected; stale/unrelated
measurements do not influence selection. Resource observations constrain the
candidate before latency ranking. Unknown constrained resources are ineligible.

The result says `Measured`, `UnmeasuredPreferred`, or `UnmeasuredEligible`.
Absent measurements never become a claimed optimal winner. Backend defaults can
preserve previously qualified geometry while making that status explicit. The
selected value is retained by physical refinement; this is not a token-path
autotuner, JIT, global cache or runtime interpreter.
