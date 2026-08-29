# Final release inventory v1

This directory documents the inert Phase 9 final-output join implemented by
`scripts/assemble-final-release-inventory.sh` and
`scripts/verify-final-release-inventory.sh`. It does not select an image,
approve a builder, create an SBOM, scan a root filesystem, sign provenance, or
grant deployment/readiness authority.

The assembler accepts one digest-suffixed 35-line input, one independently
digest-pinned external authenticator, and one new absolute output. It copies
and binds exactly eight independently approved artifacts:

- the immutable OCI image digest input;
- the final-image SBOM;
- the license inventory;
- base, builder, build, and image provenance;
- the exact AOT kernel manifest;
- the complete final-rootfs scan;
- the native/OpenAI/health/readiness/drain runtime-contract inventory; and
- the exact source/archive identity.

The source identity is itself a strict three-line
`lunaflux.source-identity.v1` record containing distinct SHA-256 identities for
the portable source archive and its exact source-file inventory.

Each artifact has a distinct approval artifact. The external authenticator is
called with:

~~~text
AUTHENTICATOR verify ROLE RELEASE_SUBJECT_SHA256 ARTIFACT ARTIFACT_SHA256 APPROVAL APPROVAL_SHA256 OCI_IMAGE
~~~

It must perform offline signature/attestation and policy verification and exit
zero only for an authentic, approved role/subject/artifact/image join. The
authenticator executable is itself selected by an external allowlist and
supplied as `ABSOLUTE_PATH#sha256=HEX`; LunaFlux does not create that trust
decision. Its exact bytes and the release tooling are preserved in the output.
Verification requires the same independently supplied authenticator identity
and replays every approval. A caller-constructed or fixture authenticator is
therefore only test evidence and cannot be promoted by this wiring.

The assembly input is exactly:

~~~text
schema=lunaflux.final-release-assembly.v1
release_subject_sha256=<64 lowercase hex>
oci_image=<untagged repository>@sha256:<64 lowercase hex>
oci_digest_source=<absolute regular file>
oci_digest_sha256=<64 lowercase hex>
oci_digest_approval_source=<absolute regular file>
oci_digest_approval_sha256=<64 lowercase hex>
sbom_source=<absolute regular file>
sbom_sha256=<64 lowercase hex>
sbom_approval_source=<absolute regular file>
sbom_approval_sha256=<64 lowercase hex>
license_inventory_source=<absolute regular file>
license_inventory_sha256=<64 lowercase hex>
license_inventory_approval_source=<absolute regular file>
license_inventory_approval_sha256=<64 lowercase hex>
provenance_source=<absolute regular file>
provenance_sha256=<64 lowercase hex>
provenance_approval_source=<absolute regular file>
provenance_approval_sha256=<64 lowercase hex>
kernel_manifest_source=<absolute regular file>
kernel_manifest_sha256=<64 lowercase hex>
kernel_manifest_approval_source=<absolute regular file>
kernel_manifest_approval_sha256=<64 lowercase hex>
rootfs_scan_source=<absolute regular file>
rootfs_scan_sha256=<64 lowercase hex>
rootfs_scan_approval_source=<absolute regular file>
rootfs_scan_approval_sha256=<64 lowercase hex>
runtime_contracts_source=<absolute regular file>
runtime_contracts_sha256=<64 lowercase hex>
runtime_contracts_approval_source=<absolute regular file>
runtime_contracts_approval_sha256=<64 lowercase hex>
source_identity_source=<absolute regular file>
source_identity_sha256=<64 lowercase hex>
source_identity_approval_source=<absolute regular file>
source_identity_approval_sha256=<64 lowercase hex>
~~~

Outputs are no-overwrite, output-adjacent staged, read-only, exactly
inventoried, symlink/special-file/hard-link free, and reverified before their
single final rename. Failure leaves no requested output. Artifact bytes are
bounded to 256 MiB each and approvals to 1 MiB each.
