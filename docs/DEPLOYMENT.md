# Reproducible OCI deployment contract

## Evidence status

LunaFlux provides a declarative runtime-image source contract, a fail-closed
build-context verifier, a Linux-only build wrapper, and static hostile-context
gates. No OCI image was built on this macOS host. This work therefore does not
prove a Linux native build, a CUDA execution path, a final rootfs scan, an SBOM,
or a reproducible final image digest.

An approved deployment must supply and retain evidence for the selected CUDA
runtime base digest, the Linux builder and buildx version/configuration, the
externally built LunaFlux binaries and libraries, the final-image digest and
scan, and physical CUDA correctness and promotion gates. An image is not
reproducible merely because its context passes this repository's verifier.

### Native Linux ABI floor

The native process owner uses GNU `posix_spawn` extensions, including
`posix_spawn_file_actions_addclosefrom_np`. A supported Linux execution host
must therefore provide glibc 2.34 or newer. Ubuntu 20.04/glibc 2.31 is not a
supported LunaFlux execution target even when it is used as a trusted build
coordinator. Native process stubs compile with `_GNU_SOURCE`, and both shipped
executables link with POSIX thread support. Treat either an implicit-declaration
diagnostic or an unresolved pthread/process-spawn symbol as a release failure.

On 2026-08-27 a temporary Ubuntu 22.04 NVIDIA host successfully ran the native
host diagnostic now named `lunaflux legacy-doctor`. It loaded driver ABI 13010 and enumerated two physical
CUDA devices. Readiness correctly remained false because no complete,
numerically validated production AOT Llama rollout unit was supplied. Device
inventory is not serving evidence.

## Authority boundary

OCI labels are diagnostic inventory only. They cannot approve a base, binary,
model, kernel artifact, descriptor, or rollout. Runtime authority remains the
two deployment-approved immutable roots and the descriptor SHA-256 supplied as
a CLI argument. The model mount marker in the image is also non-authoritative;
it only documents the required mount point.

The base reference is an independent deployment input. It must be an untagged
reference ending in `@sha256:` and exactly 64 lowercase hexadecimal characters.
The repository deliberately supplies no default or example production digest.
`metadata/base-image.ref` is separately inventoried with the release inputs.
The mandatory host verifier compares it byte-for-byte with the CLI
`BASE_IMAGE` before a container build. `Containerfile` does not and cannot
perform that independent comparison.

## Exact build context

The context contains exactly `rootfs/` and `metadata/`; symlinks, special
files, unlisted files, writable content, and unexpected paths are rejected.
All directories are mode `0555`, executables are `0555`, and other files are
`0444`.

~~~text
rootfs/
  opt/lunaflux/bin/lunaflux
  opt/lunaflux/bin/lunaflux-device-worker
  opt/lunaflux/lib/                         # exact optional runtime libraries
  opt/lunaflux/kernels/                     # one exact manifest plus AOT modules
  var/lib/lunaflux/model/.mount-contract    # diagnostic marker only
metadata/
  artifacts.sha256
  base-image.ref
  kernel-manifest.relative
  kernel-manifest.sha256
  linux-architecture
  runtime-libraries.list
~~~

Both executables and every staged library must be externally built ELF64
Linux payloads for the declared `x86_64` or `aarch64` architecture.
`runtime-libraries.list` is one comma-separated, sorted list of rootfs-relative
library paths, or the literal `none`. `artifacts.sha256` is the sorted exact
rootfs inventory in `sha256sum` format. The kernel locator is relative to
`/opt/lunaflux/kernels`; its digest must match, at least one `.cubin`,
`.fatbin`, or `.bin` module must exist, and the manifest must bind every staged
module digest and name its exact strict-relative path. This host check is only
packaging-presence evidence. LunaFlux's typed execution-manifest admission
remains the semantic authority that binds each `(path, digest)` pair to the
admitted execution contract.

No model bytes are baked into the image. Source, PTX, compilers, MoonBit tools,
Python, shells, package-manager steps, NVRTC/JIT dependencies, mutable tags,
and ambient runtime files are rejected. The verifier cannot inspect bytes that
come from the base image, so the final rootfs scan remains mandatory.

## Required build flow

First assemble and independently inventory the context on the approved Linux
release system. Do not construct it from a mutable live installation. Produce
one host-native `lunaflux` release tool with an independently approved digest,
then use the atomic materializer rather than publishing the packaging-only
assembler output. Invoking `docker buildx` directly is not the approved flow.

~~~sh
scripts/validate-oci-packaging.sh
scripts/validate-release-materialization.sh
scripts/materialize-release-bundle.sh \
  /absolute/path/to/assembly.v1#sha256=<approved-input-digest> \
  /absolute/path/to/new-release-bundle \
  /absolute/path/to/host-lunaflux#sha256=<approved-tool-digest>
# Only after an interrupted materializer has left its exact v2 claim:
scripts/recover-release-materialization.sh \
  /absolute/path/to/new-release-bundle
scripts/build-oci-image.sh \
  '<approved-registry>/<cuda-runtime>@sha256:<approved-lowercase-digest>' \
  /absolute/path/to/exact-context \
  linux/amd64 \
  /absolute/path/to/new-lunaflux.oci.tar
~~~

The materializer atomically claims the new output, assembles in private
same-filesystem scratch, and invokes `validate-materialized-release` before
installing a verifier-admitted private stage. Typed
source-root views retain no-follow staging identity while binding the copied
bytes to the final absolute model, kernel, policy, and worker labels. The
completed bundle records canonical root-free semantic evidence plus exact
tool/materializer digests. A failed semantic join removes the exact empty claim.
If the process dies during publication, the explicit recovery command accepts
only the canonical current-user-owned output with the exact mode-400 v2 claim,
an exact prefix of the six publication entries, and claim-bound prepared and
bundle inventories. It resumes and verifies that transaction. Symlinks, special
files, hard-linked metadata, unknown entries, non-prefix splits, or substituted
inventories are refusal-only. Recovery never recursively deletes the output;
CLAIMED cleanup unlinks only the authenticated regular claim and `rmdir`s the
proven-empty directory. This proves neither the tool's build provenance nor the
target image; both remain external release evidence.

The wrapper runs `verify-oci-context.sh` before buildx, confirms that the CLI
base reference exactly equals `metadata/base-image.ref`, matches the verified
architecture to the requested platform, refuses to overwrite an archive, and
passes the same base reference to the mandatory `BASE_IMAGE` build argument.
The equivalent `linux/arm64` platform requires `aarch64` context metadata.

After construction, generate the SBOM, scan the complete final rootfs and
dependency closure for shells, Python, compiler/JIT tools and writable or
unexpected paths, record base and builder provenance, and publish by the
registry-reported final image digest. Do not promote a tag. The approved image
digest is external evidence and is never inferred from an OCI label.

After those deployment-owned outputs are signed and approved, assemble the
separate `lunaflux.final-release-inventory.v1` described in
`release/final_inventory/README.md`. Supply the 35-line input and the
independently allowlisted authenticator as digest-suffixed absolute paths. The
assembler preserves the OCI digest, SBOM, license inventory, provenance, kernel
manifest, complete-rootfs scan, runtime contracts, exact source identity, and
tool identities in one deterministic no-overwrite output. Re-run its verifier
with the same independently supplied authenticator before publication. This
join records and reauthenticates external authority; it does not create or
approve any of those inputs.

## Install and start

1. Select the approved final image digest, immutable model root, baked kernel
   set, strict descriptor locator, and independently approved descriptor
   digest as one rollout unit.
2. Mount the model root read-only at `/var/lib/lunaflux/model`. Do not overlay
   or mount a writable kernel root over `/opt/lunaflux/kernels`.
3. Run with a read-only root filesystem and numeric UID/GID `65532:65532`.
   Supply no shell entrypoint and no environment variable as artifact or
   descriptor authority.
4. Run the pinned `doctor`, `plan`, and `inspect-kernels` forms before routing
   traffic. These are inert diagnostics and never grant readiness.
5. Supply a separately inventoried, read-only deployment root containing the
   fixed `lunaflux.launch.json`, plus its independently approved SHA-256. Its
   strict envelope must name the exact model, baked kernel, policy, descriptor,
   worker executable, device assignment, and listener configuration selected
   for this rollout.
6. Start only with the digest-suffixed one-argument `run` form. Route traffic
   only after that live owner reports readiness; a successful preflight alone
   is insufficient.

Conceptually, the runtime invocation has this shape; the deployment system
must substitute independently approved absolute values:

~~~sh
docker run --rm --read-only --user 65532:65532 \
  --mount type=bind,src=/approved/model-root,dst=/var/lib/lunaflux/model,readonly \
  '<approved-image>@sha256:<approved-final-image-digest>' \
  doctor /var/lib/lunaflux/model /opt/lunaflux/kernels \
  descriptor.json '<approved-descriptor-sha256>'
~~~

The five-argument diagnostic forms return `LunaModelPreflightComplete` after a
complete physical preflight without attempting activation. The production
one-argument form is
`run ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>`; it composes the live
worker, service, and listener owners and can publish readiness only after the
exact physical target and bound listener are live. This packaging contract
alone does not claim that CUDA execution or readiness has passed.

## Upgrade, drain, and rollback

Build every new runtime, base, library set, kernel set, model root, or descriptor
as a new immutable candidate. Verify it independently, start a parallel
instance, run preflights, and route traffic only after the required release
evidence passes. Never replace bytes beneath a live mount or reuse an old
context inventory for changed content.

For a healthy shutdown, stop new admissions first, retain bounded accepted
work, wait for terminal event acknowledgement, then close the worker and
container. A signal or container kill is not a substitute for instance drain.

Rollback selects the previous approved image digest, model root, baked kernel
set, descriptor locator/digest, and approval record together. Re-run preflight
against that exact unit. Never mix a previous image with a newer manifest,
descriptor, model root, or mutable tag.

## Troubleshooting

- `BASE_IMAGE is not digest pinned`, mutable-tag, digest-format, or metadata
  mismatch: reject the build and repair the independently inventoried input.
- Unexpected path, mode, symlink, checksum, library-list, or ELF architecture:
  rebuild the context from approved Linux artifacts; do not relax the gate.
- Missing or mismatched kernel manifest/module digest: republish one complete
  AOT kernel root and a new exact inventory.
- Compiler, PTX, NVRTC, Python, or shell rejection: remove the runtime toolchain
  dependency; JIT and developer fallbacks are not production paths.
- Final-image scan finds a forbidden base payload: select and approve a
  different digest-pinned runtime base. Passing context verification is not a
  waiver for base-image contents.
- Read-only or UID failures: correct ownership and the deployment mount policy;
  do not make the image, model root, or kernel root writable.
- CUDA inventory, target, driver, descriptor, or readiness failure: follow
  `docs/OPERATIONS.md`; do not select a different artifact implicitly.
