# Fused physical evidence and approval seam

This package first admits immutable physical observations for the exact two
Phase-5 fused candidates. Those observations, the candidate objects, their
typed compiled-artifact bindings, and paired benchmark qualification evidence
remain inert and report `manifest_bindable=false`.

`prepare_fused_manifest_approval_record` creates canonical bytes that bind the
exact candidate, source, recipe, CUBIN, compile receipt, compiled-binding,
physical-evidence, directory-seal, benchmark-evidence, and benchmark shape-set
digests. The record is only an input for external review and signing; it grants
no authority itself.

`admit_external_fused_manifest_approval` accepts only the opaque
`LunaAuthenticatedExternalApproval` produced by the authenticated LunaFlux
verifier. Public signed evidence is inert and cannot mint this value. LunaFlux
checks the exact record digest and envelope identities; it does not perform
Ed25519 verification or claim that it verified the detached signature itself.
Production verifier keys now enter only through the bounded, authenticated,
one-shot startup handoff. The parent verifies the record, wipes the deployment
key, and gives a child only an exact launch-bound attestation; the deployment
key never crosses into worker execution. Without that authenticated startup
input, this promotion route remains unavailable rather than caller-asserted.

The resulting `ApprovedFusedManifestArtifacts` value has exactly one
production-manifest projection in this package. The current record binds the
canary-bearing residual/RMSNorm qualification artifact, so that projection now
fails closed at the production-artifact gate. A successor record must bind the
distinct canary-free production ABI and its physical result before a manifest
can be produced; an existing qualification signature cannot be replayed as
production authority. A successfully projected manifest retains the exact
approval, physical, benchmark, compiler, target, launch, symbol, ABI, artifact,
purpose, and typed standalone-fallback identities. The separately gated
`luna_fused_artifact_admission` package may turn only that opaque projection
into an immutable in-memory bundle or exact catalog.

Neither package loads a module, opens a device, executes a kernel, or serves
traffic. Device-step preparation has a grouped residual/RMSNorm replacement
contract, but production remains unavailable until the successor physical and
external approval record exists. Qualification keeps its dispatch canary;
production accepts only the canary-free ABI.
