# Digest-pinned runtime descriptor admission

`runtime/descriptor_file` owns one strict startup-only JSON boundary for the
single-device dense-Llama runtime recipe. The descriptor is authenticated by an
independently supplied lowercase SHA-256 digest; a digest stored inside the
same file would not be authority.

Model configuration and safetensors descendants resolve only beneath a
caller-owned model root. The execution manifest and its AOT modules resolve
only beneath a separate caller-owned kernel root. Both canonical absolute
labels are checked against their exact pinned capabilities before any file is
opened. The loader retains neither capability.

Successful admission composes existing validators and builders into immutable
model metadata, a paged semantic plan, complete weight-file inspection,
canonical KV layout, production paged-execution admission, admitted bootstrap
manifest, canonical bootstrap source, startup contract, and inert
`DeviceWorkerPlan`. It opens no device and owns no native resource. Deployment
remains responsible for signature verification and immutable mount selection;
the descriptor policy field cannot grant that authority.

The eager-only descriptor schema `lunaflux.runtime.v1` remains supported and
represents graph-memory accounting as explicitly absent. Strict
`lunaflux.runtime.v2` adds one required positive `max_graph_capture_bytes`
ceiling. It is admitted only when authenticated capture-safe graph metadata
declares a positive aligned upper bound beneath that ceiling; a ceiling cannot
manufacture missing graph evidence. Unknown, duplicate, missing, malformed,
over-depth, and over-byte inputs fail closed. All file owners close
before admitted bytes or typed evidence can be published, while both roots
remain caller-owned for exact aggregate cleanup.

A separate `lunaflux.runtime.i8.v2` loader admits the production dense-Llama
symmetric-I8 weight-only recipe. It independently pins the numeric-weight
artifact, requires the exact 8.9, 9.0, or 12.0 device-fact envelope, derives the
numeric capability from those inert facts, and composes manifest v4 with
bootstrap v3 and worker-source v6. Its opaque admission owns no root, device,
allocation, executor, or readiness authority. Physical readiness therefore
remains false until the admitted runtime is rebuilt and validated on the
assigned CUDA machine.
