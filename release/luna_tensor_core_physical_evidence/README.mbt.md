# LunaTile tensor-core physical evidence

This release boundary is prepared to admit only the exact sealed `sm120` BF16
WMMA campaign and bind it back to the deterministic LunaTile candidate, serial
oracle, and SIMT fallback. Admission positionally checks the 67-field result,
critical `FILES.sha256` entries, the result-bound inner manifest digest, and an
out-of-band digest for `OUTER_SEAL.sha256`; the outer seal binds both
`FILES.sha256` and `RESULT.txt` without a self-hash cycle.

The record must retain deterministic serial and tensor-core CUBIN pairs, exact
device/compiler/tool identities, matching positive SASS instruction counts
from `cuobjdump` and `nvdisasm`, bounded registers/shared/stack/local/spills,
independent CPU and serial-CUDA numeric comparisons, uniquely clean
memcheck/racecheck/initcheck logs, and zero cleanup balance. The separate
evidence-aware lowering join re-emits the deterministic candidate and rejects
any serial or SIMT fallback substitution.

This package does not read files, invoke CUDA tools, load a module, or grant
manifest, deployment, runtime, or promotion authority. The evidence-free
`RequireExternallyQualifiedTensorCore` specialization path remains rejecting.
No campaign record has been produced on NVIDIA hardware yet, so there is no
physical tensor-core qualification to admit or promote.
