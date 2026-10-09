///|
fn official_audio_vae_latent_means() -> Array[Double] {
  [
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
  ]
}

///|
fn official_audio_vae_latent_standard_deviations() -> Array[Double] {
  [
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
  ]
}

///|
fn audio_vae_f32(values : Array[Double]) -> ReadOnlyArray[Float] {
  ReadOnlyArray::from_array(values.map(Float::from_double))
}

///|
fn validate_official_audio_vae_scaling(
  object : Map[String, Json],
) -> Unit raise MiniMaxH3ConfigError {
  exact_number_array(object, "latents_mean", official_audio_vae_latent_means())
  exact_number_array(
    object,
    "latents_std",
    official_audio_vae_latent_standard_deviations(),
  )
}

///|
fn official_audio_vae_config() -> MiniMaxH3AudioVaeConfig {
  {
    architecture_schema_version: 1,
    encoder_dimension: 64,
    encoder_rates: ReadOnlyArray::from_array([2, 4, 4, 5, 5]),
    latent_dimension: 2048,
    latent_channels: 32,
    decoder_dimension: 1024,
    decoder_rates: ReadOnlyArray::from_array([5, 5, 2, 2, 2, 2, 2]),
    decoder_kernel_sizes: ReadOnlyArray::from_array([9, 9, 4, 4, 4, 4, 4]),
    attention_head_count: 8,
    resblock_kernel_sizes: ReadOnlyArray::from_array([3, 7, 11]),
    resblock_dilation_sizes: ReadOnlyArray::from_array([
      ReadOnlyArray::from_array([1, 3, 5]),
      ReadOnlyArray::from_array([1, 3, 5]),
      ReadOnlyArray::from_array([1, 3, 5]),
    ]),
    sampling_rate: 32000,
    output_channels: 2,
    latent_rate: 40,
    latent_means: audio_vae_f32(official_audio_vae_latent_means()),
    latent_standard_deviations: audio_vae_f32(
      official_audio_vae_latent_standard_deviations(),
    ),
  }
}
