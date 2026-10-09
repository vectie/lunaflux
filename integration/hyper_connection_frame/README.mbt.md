# Executable residual-stream envelope

`HyperConnectionPrecision` is a pure numerical/memory plan. CUDA lowering
composes function/control projection and collapse, positive Sinkhorn, weighted
input RMSNorm, and residual publication. The frame owner allocates six exact
intermediates at startup and binds four functions from a borrowed AOT module.

Preparation returns three prefix launches and one suffix launch. A numerical
branch borrows them into its own prepared execution queue, using `branch_input`
and `branch_output` as intermediate I/O. This adds no queue or completion event
between the envelope and branch. Close the borrowing queue before the frame;
close releases functions and buffers deterministically, including partial setup.

The rounded-BF16 contract consumes `combination[source,destination]`, rounds
controls and branch/residual products at their declared boundaries, and publishes
BF16. The ordered-F32/single-round contract consumes
`combination[destination,source]`; it is a different numerical contract, not an
interchangeable speed option. The GLM adapter selects the former.

No model names, request JIT, token-step allocation, filesystem checks or
diagnostic state readbacks occur in this generic owner. The existing control
projection and Sinkhorn lowerings are correctness-first, serial within a row;
this feature does not claim a throughput-optimal schedule.
