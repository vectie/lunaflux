# Network process settings

`validate_process` checks the operator's node label and exact NCCL transport
selection once before a remote worker opens resources. Set `LUNAFLUX_NODE_ID`
to the declared label, `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, and
`NCCL_IB_HCA==mlx5_0` (the value starts with `=`; replace the interface).
The environment must remain fixed for the lifetime of the process. A process
must not initialize NCCL under different settings before starting its rank.

This is configuration validation, not peer authentication or measurement of
RoCE availability. Physical link qualification remains separate.
NCCL selection semantics follow the [NVIDIA environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
