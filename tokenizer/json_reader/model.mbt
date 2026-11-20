///|
fn parse_model(
  value : Json,
  specials : Map[Int32, AddedSpecial],
  digest : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
  profile : TokenizerJsonProfile,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  guard value is Object(model) else {
    raise InvalidField(component=Root, field=ModelField, index=0)
  }
  validate_fields(model, Model, [
    "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix",
    "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges",
  ])
  if require_string(model, "type", Model, TypeField, 0) != "BPE" {
    raise UnsupportedSemantics(component=Model, feature=ModelKind)
  }
  require_null(model, "dropout", Model, Dropout, MergeDropout)
  match profile {
    NfcSingleDigitJson =>
      for
        (name, field, feature) in [
          (
            "continuing_subword_prefix",
            ContinuingSubwordPrefix,
            ContinuingSubwords,
          ),
          ("end_of_word_suffix", EndOfWordSuffix, WordSuffixes),
        ] {
        match require_field(model, name, Model, field) {
          Null | String("") => ()
          _ => raise UnsupportedSemantics(component=Model, feature~)
        }
      }
    Qwen3Json | Qwen35Json => {
      if require_string(
          model,
          "continuing_subword_prefix",
          Model,
          ContinuingSubwordPrefix,
          0,
        ) !=
        "" {
        raise UnsupportedSemantics(component=Model, feature=ContinuingSubwords)
      }
      if require_string(model, "end_of_word_suffix", Model, EndOfWordSuffix, 0) !=
        "" {
        raise UnsupportedSemantics(component=Model, feature=WordSuffixes)
      }
    }
    _ => {
      require_null(
        model,
        "continuing_subword_prefix",
        Model,
        ContinuingSubwordPrefix,
        ContinuingSubwords,
      )
      require_null(
        model,
        "end_of_word_suffix",
        Model,
        EndOfWordSuffix,
        WordSuffixes,
      )
    }
  }
  let (unknown_token_id, bos_token_id) = match profile {
    MiniCPM5Json => {
      if require_string(model, "unk_token", Model, UnknownToken, 0) != "<unk>" ||
        require_bool(model, "byte_fallback", Model, ByteFallback, 0) ||
        require_bool(model, "fuse_unk", Model, FuseUnknown, 0) ||
        optional_ignore_merges(model) {
        raise UnsupportedSemantics(
          component=Model,
          feature=UnknownTokenFallback,
        )
      }
      // The byte-complete alphabet is checked by TokenizerSpec. Consequently
      // this declared unknown token cannot be reached by UTF-8 byte encoding.
      (-1, 0)
    }
    ByteLevelJson
    | Qwen3Json
    | Qwen35Json
    | DigitTripletJson
    | NfcSingleDigitJson
    | OrderedScriptJson => {
      require_null(
        model,
        "unk_token",
        Model,
        UnknownToken,
        UnknownTokenFallback,
      )
      if require_bool(model, "fuse_unk", Model, FuseUnknown, 0) {
        raise UnsupportedSemantics(component=Model, feature=UnknownTokenFusion)
      }
      if require_bool(model, "byte_fallback", Model, ByteFallback, 0) {
        raise UnsupportedSemantics(
          component=Model,
          feature=ByteFallbackBehavior,
        )
      }
      (-1, -1)
    }
    SentencePieceJson(bos_token_id~) => {
      let unknown = require_string(model, "unk_token", Model, UnknownToken, 0)
      if unknown != "<unk>" ||
        !require_bool(model, "fuse_unk", Model, FuseUnknown, 0) ||
        !require_bool(model, "byte_fallback", Model, ByteFallback, 0) {
        raise UnsupportedSemantics(
          component=Model,
          feature=UnknownTokenFallback,
        )
      }
      match model.get("ignore_merges") {
        None | Some(Null) | Some(False) => ()
        _ =>
          raise UnsupportedSemantics(
            component=Model,
            feature=IgnoreMergeBehavior,
          )
      }
      let mut unknown_token_id = -1
      for id, special in specials {
        if special.content == unknown {
          unknown_token_id = id
        }
      }
      if unknown_token_id < 0 {
        raise UnsupportedSemantics(
          component=Model,
          feature=UnknownTokenFallback,
        )
      }
      (unknown_token_id, bos_token_id)
    }
  }
  let ignore_merges = match profile {
    ByteLevelJson
    | Qwen3Json
    | Qwen35Json
    | DigitTripletJson
    | MiniCPM5Json
    | NfcSingleDigitJson
    | OrderedScriptJson => optional_ignore_merges(model)
    SentencePieceJson(..) => false
  }
  let vocabulary = parse_vocabulary(
    require_field(model, "vocab", Model, Vocab),
    specials,
    limits,
    profile,
  )
  let merges = parse_merges(
    require_field(model, "merges", Model, Merges),
    vocabulary,
    limits,
    profile,
  )
  match profile {
    MiniCPM5Json =>
      @tokenizer.TokenizerSpec::new_minicpm5_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
      ) catch {
        error => raise Validation(error)
      }
    OrderedScriptJson =>
      @tokenizer.TokenizerSpec::new_ordered_script_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
        ignore_merges~,
      ) catch {
        error => raise Validation(error)
      }
    DigitTripletJson =>
      @tokenizer.TokenizerSpec::new_digit_triplet_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
        ignore_merges~,
      ) catch {
        error => raise Validation(error)
      }
    ByteLevelJson =>
      @tokenizer.TokenizerSpec::new(
        vocabulary.entries,
        merges,
        digest,
        limits,
        ignore_merges~,
      ) catch {
        error => raise Validation(error)
      }
    Qwen3Json | NfcSingleDigitJson =>
      @tokenizer.TokenizerSpec::new_qwen3_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
        ignore_merges~,
      ) catch {
        error => raise Validation(error)
      }
    Qwen35Json =>
      @tokenizer.TokenizerSpec::new_qwen35_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
        ignore_merges~,
      ) catch {
        error => raise Validation(error)
      }
    SentencePieceJson(..) =>
      @tokenizer.TokenizerSpec::new_sentencepiece_bpe(
        vocabulary.entries,
        merges,
        digest,
        limits,
        unknown_token_id~,
        bos_token_id~,
      ) catch {
        error => raise Validation(error)
      }
  }
}
