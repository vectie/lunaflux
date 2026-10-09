# Two-node network declarations

Immutable, authority-free RoCE topology inputs. Node labels qualify CUDA
ordinals, so both workers can legitimately use ordinal zero. Distinct node
labels and homogeneous exact targets are required. This package performs no
probe, authenticates no machine, and does not create an `AdmittedLocalTopology`.
Live admission must verify node/device/NIC identity and actual connectivity.

```mbt check
///|
test {
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let ranks = [
    for name in ["spark-a", "spark-b"] => {
      @network_topology.NetworkRank::new(
        node=@network_topology.NodeIdentity::new(name),
        device_ordinal=0,
        target~,
        physical_memory_bytes=128000000000L,
        rdma_interface="mlx5_0",
      )
    }
  ]
  let topology = @network_topology.TwoNodeTopology::new(ranks[0], ranks[1])
  assert_eq(topology.rank(1).unwrap().device_ordinal(), 0)
}
```
