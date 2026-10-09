# DeepSeek V4 token-hash operand planning

This narrow integration package joins three already-authenticated boundaries:
live DeepSeek checked-I32 token-hash sidecars, their separately allocated and
validated segmented-device layout, and a family-neutral advanced-decoder
token-hash CUDA source candidate. It is intentionally outside the DeepSeek
device adapter and generic device/kernel cores.

One inert plan selects `TokenHashTableInput` operand ordinal 2 for exactly one
official hash layer 0 through 2. Admission authenticates the shared model and
requirements identity, exact `TokenHashRouting(3,129280,6,experts)` operation,
I32 candidate contract, operand role/3,102,720-byte extent/4-byte alignment,
and the matching sidecar region, tensor ordinal, and aligned device offset.
The plan digest also binds candidate source/recipe digests, sidecar source and
layout digests, source manifest digest, and the selected table digest.

The operand-plan result is `InertUnqualified`. Its `require_launch_authority`
always fails because the candidate contains source only; no compiled module,
loader authority, or physical qualification has entered that state. Scheduler
and generic device packages must not import this integration package.

The next startup-only state, `DeepSeekV4TokenHashArtifactBinding`, joins that
operand plan to an inert `AdvancedDecoderArtifactAdmission`. It authenticates
the same candidate source and recipe identities, target, requirement ordinal,
operation, capability, launch dimensions and every operand, then resolves the
admitted entry-point symbol and digest-verified module. Its canonical digest
binds the operand plan, candidate recipe, admission, module, entry point,
symbol, launch, and selected device offset.

This remains evidence, not a prepared CUDA launch. The artifact admission does
not prove that its module was produced from the candidate recipe, and it grants
no module-loader or execution authority. Compilation provenance, loader
authority, launch authority, and physical qualification therefore each have a
separate typed `require_*` failure. A future owner must supply an authenticated
offline producer policy, live-device target evidence, loaded module and function
lifecycle ownership, complete allocation regions for the other three operands,
and correctness qualification before launch preparation is possible.

`DeepSeekV4TokenHashCompileEvidenceBinding` adds one narrower join to the
family-neutral deterministic offline-compile receipt boundary. It requires the
receipt's source and recipe digests, complete compiler policy, target, function
symbol, module digest, and byte count to equal the candidate and admitted
artifact exactly. The canonical binding digest commits to both upstream
evidence digests and all joined identities.

This additional state is still `InertSelfConsistentUnqualified`: the canonical
receipt is supplied alongside its module snapshots and therefore authenticates
self-consistency, not a named producer or an observed compiler execution.
Compilation provenance and every loader, launch, and physical authority remain
explicit typed failures. A trusted offline producer/signature policy, loader
lifecycle owner, remaining operand regions, and qualification evidence are
still required before launch preparation.

`DeepSeekV4TokenHashLaunchBinding` completes the four-operand inert metadata
join. Callers provide bounded allocation-relative descriptors for
`StepCountsInput`, `TokenIdsInput`, and `RoutingIndicesOutput`; the package
reconstructs `TokenHashTableInput` only from the exact previously validated
sidecar layout and operand plan. The shared region planner enforces ABI order,
roles, byte counts, alignments, allocation capacity, and pairwise non-overlap.

The launch-binding digest additionally commits to the model and requirement
identity, maximum-token and routing geometry, candidate source and recipe,
complete compiler policy and target, artifact module/entry point/symbol/launch
dimensions, and compile-evidence identities. It remains
`InertCompleteUnqualified`: descriptors are not live pointers, and loader,
launch, and physical-qualification authority are separate typed failures.
