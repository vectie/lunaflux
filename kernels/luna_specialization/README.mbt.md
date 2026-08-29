# Typed LunaTile offline specialization

This package specializes one admitted model-plan operation using only an exact
`ModelPlan`, `StaticDevicePlan`, `DeviceExecutionProfile`, resolved kernel
catalog, admitted LunaTile program, and typed offline policies. It verifies
identity, target, catalog version, operation order and binding, workspace,
profile shape, content-addressed AOT family, and one opaque exact LunaTile
residual-add contract before publishing a record. It binds the ModelPlan's two
ordered Activation inputs and one output to the exact execution-profile
references, widths, token rows, and BF16 byte geometry. Generic programs and
all other model-operation families fail closed. The current unquantized BF16
path rejects FP16 or integer LunaTile storage before record emission.

The canonical record binds checked 64-bit weight, token, activation, output,
and workspace regions; exact RoPE and RMS epsilon bit patterns; the BF16
unquantized policy; compiler identity and finite flags; numerical policy;
stable entry-point identity; LunaTile and CUDA-AOT-plan-input digests; and four
content-addressed evidence references. Reassociation cannot claim bit-exact
operation order.

Specialization is an offline deterministic transformation. The record exposes
no generated source, compiler handle, model loader, device context, module,
request, or runtime JIT path. Quantized and non-BF16 specialization remain
unsupported until their typed layouts and codebooks exist.
