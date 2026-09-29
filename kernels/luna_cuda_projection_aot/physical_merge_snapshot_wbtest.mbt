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
      #|  "68bac0af51271d945a347853cf5014a0dac71863067e7b7687006db6b50669c9",
      #|  "d7b37eca2c764458d43eeb15fddc873eff55bfa58ba452e28130bab01cad4875",
      #|  "610961c2b8c6dddbd77b07dece8b96090f2c44e0342b855c441e3b71ff72c0e1",
      #|  "a62b2976f07ef52363ec3444c3187095760ca960560f20fe5d583c942e197a6f",
      #|  "38d5ef2769a21118cd8d6adcfe2a28bbf3d22d1a2007fa73ccaad0ec358945d5",
      #|  "c4bcc0ee3b44653470aa7e4df3081724217e8300afef13f4641b9211fc45c722",
      #|  "9730dbb9e4f2b81b8b6a22eed3115c86d2b3eba74f1637bc4229a7c230b342a1",
      #|  "cc47e3ef950c51af030188b4d6887a132604d227eff9152036fbb05f77e7fafe",
      #|  "870604879503b06abc32191719f1a331304796dc7ecc0f3587be17718cc73f5b",
      #|  "e5cd234a5a45b84dabac41c825f0c3a4333ebabb012a84ab0ca9597628c2f3fc",
      #|  "448ed5485e271d53158f0ee0f08ff1903a046a25a484e34b9b141ba65a785ae6",
      #|  "48adcf39545e4eccd4a60a5aa2371901d3ba8acc1af0c0b9dbfc43a28cd526a0",
      #|  "646c96d7da09d3be91f5df61a1911d1bc7d46e0ea5d9c51e2c40be9e6bcf0569",
      #|  "198e63b2847fa979664f00ee92e47396d39bdd9991876edc899e9e09c179c1d4",
      #|  "f2063feeb0c3220938889a28543573d69a4092e450cad494ee50128397213310",
      #|  "08f12402067be86fd0e72a92052b684c722c667bfcedc568ce79abe252e24e81",
      #|  "0c793e9c31aad89d46c988d85a1601fb1b0ecb8a0e8ac90a0b90f17416c122a5",
      #|  "cd15c50ebab50cf9a845cb26388d918d99793e77a02218ef2ff5c96bc9718b9b",
      #|  "ffe3de2fcd1b5d454159e9629b727d0c0e9ead72c3293e5ee0a987fdf950c9c1",
      #|  "9cbca65772a2d9f48ae2fbb49383dab5b0f3a59985c9b423d850c38441739186",
      #|  "0553377ec2d993db23244addcff24c81899b486d840dd08e5e1bd60f9d1a34f1",
      #|  "8431b6d7e9f344f36f8f5b6f52819f38027dbdab97a00b5b73a2d6a4842aa35c",
      #|  "de3788360728b60da67c128df8d5a2697c63d579532e3d2db69d0ef4cbcbff38",
      #|  "a501f0b9ba938f8b277f774811ba8035b602ea44d3cdc26a50116903371ed237",
      #|]
    ),
  )
}
