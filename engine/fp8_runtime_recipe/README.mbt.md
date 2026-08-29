# FP8 runtime recipe

This package is the last pure Phase-8 composition boundary before device
resource construction. It joins one admitted dense-Llama FP8 plan, one
authenticated numeric-file inspection, admitted AOT artifacts, observed
numeric capability, and the complete ordered dynamic-launch ABI set.

Every FP8 weight operand and scale-tensor launch region must exactly equal the
corresponding offset and length in the inspected numeric layout. Their logical
storage identity is derived from the pinned numeric artifact digest. Every
activation-scale and kernel-workspace region must name one distinct,
fixed-capacity future workspace allocation.

Weight and scale claims also use the layout's exact arena alignment, not a
caller-selected weaker alignment. Canonical identity includes that alignment,
the complete ordered inspected layout, every weight and scale region, and all
workspace regions. Its configured byte ceiling is enforced before every
canonical-output growth.

The result is deliberately inert. This package owns no device allocation or
executable module/function and cannot by itself create an executor, worker,
route, or readiness claim.

The wider runtime now has an authenticated FP8 route. Exact launch recipe
`DenseLlamaFp8ReusablePagedAotApprovedV9` selects the strict schema-v5
descriptor, opaque external CUBIN approval, FP8 instance admission, child-side
reconstruction, and the common rooted service/lifecycle owner used by the
one-argument CLI. Those owners remain outside this pure package. Their focused
software and cleanup gates do not constitute physical FP8 evidence: no
supported FP8 GPU has executed the route, and numerical, sanitizer/leak, soak,
accuracy, memory, performance, readiness, and promotion gates remain open.

## Staged recipe v2

`Fp8RuntimeRecipeV2` is a separate authority domain. It accepts only
`Fp8StartupAdmissionV2` and complete ordered `Fp8StagedDynamicLaunchAbi`
values; v1 startup and launch values are not valid arguments. The startup
identity retains the exact model-plan digest, including simple or compound
activation-scale policy and its declared workspace semantics.

Each staged launch reauthenticates its plan, profile, operation kind,
execution, module, entry point, typed stages, FP8 weights, and scalar scales.
V2 accepts only the ABI's authenticated catalog-v4 paged launch authority; its
source version, launch dimensions, and complete ordered raw operand table are
validated and bound directly into the recipe canonical identity. Every weight
must be immediately paired with its exact `WeightScaleInput`, and Workspace
must remain the final exact raw operand.
The first whole `Workspace` derives one future activation-workspace arena
identity and capacity. Every later launch must name that same arena; overlapping
4- or 8-byte Workspace regions are allowed only in manifest order. The v2
result remains inert and grants no executor, CLI, readiness, or physical claim.

## Reusable paged recipe v3

`Fp8ReusablePagedRecipeV3` derives the reusable PagedV4 launch set from v2,
the exact paged profile, and canonical KV layout. It removes the v2 concrete
row shape while retaining operation order, launch authority, raw operands,
workspace alignment, deterministic arena offsets, profile ceilings, and the
model/KV identity. Every live frame still requires separate device-step
admission before enqueue. The recipe is inert evidence consumed by the wider
schema-v5 release route; it does not itself own CUDA resources or claim a
physical result.
