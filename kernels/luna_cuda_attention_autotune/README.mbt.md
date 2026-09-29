# CUDA attention autotune profiles

This backend package stores immutable offline measurements. Profiles are pure
functions from a complete attention problem and portable capabilities to
autotune records. The generic LunaTile compiler owns selection; this package
contains no runtime benchmark, global cache, model-family branch, or request
path work.

Fresh exporters use `cuda_architecture_target(major, minor)`, an explicitly
uncalibrated architecture namespace. Historic named-board profiles remain
distinct; exporting for sm121 never impersonates the sm120 RTX 5060 Ti timing
profile. Measured resource inputs additionally bind exact device, toolchain,
shape and source identities. An empty timing table is not a measured win.
