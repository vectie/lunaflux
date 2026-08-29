# Pinned launch-file admission

`load_argument` accepts exactly one canonical absolute deployment-root label
followed by `#sha256=` and 64 lowercase hexadecimal digits. It opens only the
fixed `lunaflux.launch.json` descendant. The independent suffix authenticates
the immutable snapshot; both the file and launch-root authorities are closed
before JSON claims are published.

The strict `lunaflux.launch.v1` envelope remains legacy-only. The separate
`lunaflux.launch.v2` envelope additionally requires one closed, exact runtime
recipe: legacy paged AOT v5 or production symmetric-I8 paged AOT v6. I8 v6
rejects external Luna approval because its worker source cannot carry that
claim; it is never silently discarded. Both schemas bind three independent
roots, the runtime descriptor, instance policy, and worker executable.
Returned evidence is inert and owns no filesystem, process, socket, scheduler,
or device authority.
