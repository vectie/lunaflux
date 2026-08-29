# LunaTile tensor-core CUDA probe

This test-only package exports the exact `sm120` BF16 `m16n16k16` WMMA source,
its canonical candidate, the generic serial CUDA oracle, and the bound SIMT
fallback identity. The physical runner is prepared to compile serial and
tensor-core sources twice, require byte-identical CUBINs, compare an independent
ordered-F32 CPU oracle plus the serial CUDA oracle, obtain matching allowed SASS
instruction families/counts from `cuobjdump` and `nvdisasm`, inspect resource
bounds, and run memcheck, racecheck, and initcheck before publishing a
non-circular sealed result.

The local fixture, numeric logic, fake-tool hostile campaign, and evidence
admission tests pass. No NVIDIA campaign has run for this candidate, no SASS or
numeric observation is claimed, and neither the probe nor its evidence grants
manifest, runtime-loading, performance, or promotion authority.
