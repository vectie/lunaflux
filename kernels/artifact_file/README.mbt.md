# Bounded AOT artifact-file admission

`kernels/artifact_file` is the filesystem trust boundary for production AOT
CUDA artifacts. A caller supplies one approved root, a lexical manifest path
identifier, and an independently obtained manifest SHA-256. The exact JSON
manifest pins the model content and plan identities, exact device target,
catalog version, module digests and relative path identifiers, and stable
family/entry-point identities with bounded CUDA symbols.

The same parser and same-handle loader admit both stateless catalog-v1 launch
contracts and paged catalog-v3 launch contracts. The contract set supplies the
expected model, target, and catalog evidence; manifests cannot select or
downgrade that evidence.

Every path component is checked without following its final symlink, every
opened object must be a regular file, and all sizes are checked before module
snapshot allocation. The complete manifest and module snapshots are hashed
from the same opened handles. Module files are opened together and their
aggregate size is proven before any module-sized host allocation. Admission
then delegates required-module, required-entry-point, symbol, and content
semantics to the matching `kernels/artifact` admission function; this package
does not duplicate that authority or expose a bundle constructor.

The deployment contract requires an approved, read-only, stable artifact
mount. MoonBit's portable filesystem API does not expose atomic
`openat(O_NOFOLLOW)` plus handle-relative traversal, so the component checks
and opened-handle checks do not prove race exclusion in an adversarial writable
directory. There is no executable metadata, search path, environment lookup,
cache, compiler, JIT, or fallback channel.
