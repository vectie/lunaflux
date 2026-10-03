///|
test "physical refinement preserves full projection family source" {
  let digests : Array[String] = []
  for family in [QkvProjection, DenseProjection, GatedMlp, LanguageModelHead] {
    for width in [256, 1024] {
      let parameters : ProjectionParameters = {
        family,
        input_width: width,
        output_width: if family == QkvProjection {
          width * 4
        } else {
          width
        },
        query_width: if family == QkvProjection {
          width * 2
        } else {
          0
        },
        intermediate_width: if family == QkvProjection {
          width
        } else if family == GatedMlp {
          width * 3
        } else {
          0
        },
      }
      let runtime = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [],
      )
      for rows in [32, 256, 1024] {
        let source = row_variant_primary_test_source(
          parameters,
          runtime.dispatch_plan(rows),
          maximum_tokens=rows,
        )
        digests.push(projection_sha256(@utf8.encode(source)))
      }
    }
  }
  debug_inspect(
    digests,
    content=(
      #|[
      #|  "056cd88e951dc4f79d1c5c0bb7a5c07c2211bba03d7dafa2591a526337faadc1",
      #|  "ff0c29a6cfab4dd86a3ead705199d1317f1143a0bb30480ff10c775c676782d9",
      #|  "5faf53d6bb6461231d7f7203a9c829b1350f58bd11f6badb8de46752450c3644",
      #|  "fb921a10a250dc8eb824ea10f317b6445ea2c0a0987c2cddd8b3c8528efdc08f",
      #|  "581a2c7450fa57b769ab004142e666dddcb1eda04a7026c56d860c078d1fa054",
      #|  "59a74012d072f3fad732552758b9c89671fe356f10a9ed8f75ae0eea6031d83a",
      #|  "f5e60b5cb9fb4a425a669edb32b21a11af3dc449f0591789ca7e6ca8245d7bd1",
      #|  "e62a3c25e9f62e33832579bbd019680f77ea92b155b3d247628a3733f4f5b75c",
      #|  "3ee271a2eb9a5b2a921b1d60d83c643b03cc152ad284c43278d073ecb79c11a9",
      #|  "49927f3349ba883bb41d925124e753192daacef97eb7e1659c2abd23d625482d",
      #|  "3814d6af1d67a0bb5ec804f0c77f1d250c0d0d3b61fa54bb6c5faf9e4363c2e1",
      #|  "c80f9213e49235a43c15b0cbaac8d544e65e9708b215973fd00636ef19e9c458",
      #|  "646c96d7da09d3be91f5df61a1911d1bc7d46e0ea5d9c51e2c40be9e6bcf0569",
      #|  "198e63b2847fa979664f00ee92e47396d39bdd9991876edc899e9e09c179c1d4",
      #|  "f2063feeb0c3220938889a28543573d69a4092e450cad494ee50128397213310",
      #|  "08f12402067be86fd0e72a92052b684c722c667bfcedc568ce79abe252e24e81",
      #|  "0c793e9c31aad89d46c988d85a1601fb1b0ecb8a0e8ac90a0b90f17416c122a5",
      #|  "cd15c50ebab50cf9a845cb26388d918d99793e77a02218ef2ff5c96bc9718b9b",
      #|  "841b7327876ca71313ccdb066b65874f76ad989cf65e9021e7666e7f42255b07",
      #|  "5b4717598e72705a29e110554ed062ea84b842956a720c5e4d34c874f1e081a7",
      #|  "f951f87ea3286d640eae011a8c94608a9fe734cd8eae4fb68e6e66036a4d7821",
      #|  "af919030e872748e31b489b3af0e1a7c76abff671678fbef505d8849f2f29aa3",
      #|  "8729877ec1d8672bb5c60b5ef229a2ed664e13e5982558e8cfb47945cf7873bc",
      #|  "5ac7fb703f78827f64ed6b85c4d5cf590657665d481a0f9014f56e8bcf109f86",
      #|]
    ),
  )
}
