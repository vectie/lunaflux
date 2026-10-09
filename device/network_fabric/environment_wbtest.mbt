///|
test "network startup requires explicit IB and an exact single HCA" {
  validate_values(
    "spark-a",
    "mlx5_0",
    Some("spark-a"),
    Some("IB"),
    Some("0"),
    Some("=mlx5_0"),
  )
  for
    hca in [
      None,
      Some("mlx5_0"),
      Some("=mlx5_1"),
      Some("=mlx5_0,mlx5_1"),
      Some("^mlx5_1"),
    ] {
    try
      validate_values(
        "spark-a",
        "mlx5_0",
        Some("spark-a"),
        Some("IB"),
        Some("0"),
        hca,
      )
    catch {
      InterfaceMismatch => ()
      _ => fail("unexpected rejection")
    } noraise {
      _ => fail("ambiguous or substituted HCA accepted")
    }
  }
  for transport in [None, Some("Socket"), Some("")] {
    try
      validate_values(
        "spark-a",
        "mlx5_0",
        Some("spark-a"),
        transport,
        Some("0"),
        Some("=mlx5_0"),
      )
    catch {
      TransportMismatch => ()
      _ => fail("unexpected rejection")
    } noraise {
      _ => fail("fallback transport accepted")
    }
  }
  try
    validate_values(
      "spark-a",
      "mlx5_0",
      Some("spark-b"),
      Some("IB"),
      Some("0"),
      Some("=mlx5_0"),
    )
  catch {
    NodeMismatch => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("foreign node accepted")
  }
  try
    validate_values(
      "spark-a",
      "mlx5_0",
      Some("spark-a"),
      Some("IB"),
      Some("1"),
      Some("=mlx5_0"),
    )
  catch {
    TransportMismatch => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("disabled RDMA accepted")
  }
}
