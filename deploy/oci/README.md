# LunaFlux OCI source contract

This directory is declarative packaging input, not a checked image or release
artifact. `Containerfile` accepts one mandatory `BASE_IMAGE`; the repository
does not invent or select a CUDA runtime digest.

The build context is assembled externally and verified with
`scripts/verify-oci-context.sh`. Release construction must use the Linux-only
`scripts/build-oci-image.sh` wrapper so that verification, including exact
equality between its CLI `BASE_IMAGE` and independently inventoried
`metadata/base-image.ref`, completes before buildx. Its exact layout is documented in
`docs/DEPLOYMENT.md`. The context contains Linux-native LunaFlux executables,
any additional runtime libraries, one complete content-addressed AOT kernel
root, and diagnostic metadata. It contains no model payload, source tree,
compiler, package manager, shell, Python runtime, PTX, or JIT library.
The host verifier's manifest path/digest checks are packaging-presence evidence;
typed LunaFlux manifest admission remains semantic `(path, digest)` authority.

The image runs as numeric UID/GID `65532:65532`. The baked kernel root is
`/opt/lunaflux/kernels`; the model root is an external read-only mount at
`/var/lib/lunaflux/model`. The container root filesystem must also be launched
read-only.

OCI labels are never trust authority. Runtime authority remains the caller's
approved model/kernel roots and independently supplied descriptor SHA-256.
Linux image construction, final-rootfs/SBOM scanning, CUDA execution, and
physical promotion are external gates and were not run on the macOS host. No
reproducible final image digest is claimed without the approved base, builder,
and final-image scan evidence.
