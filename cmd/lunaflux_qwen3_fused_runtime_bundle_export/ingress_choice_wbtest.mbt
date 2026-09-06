///|
fn ingress_comparison_arguments(
  full : String,
  partial : String,
  unfused : String,
) -> Array[String] {
  let args = runtime_bundle_test_arguments("5")
  for
    item in [
      "--ingress-comparison", full, partial, unfused, "3", "256", "1", "1", "128",
      "0", "/tmp/postprocess.cubin",
    ] {
    args.push(item)
  }
  args
}

///|
test "evaluation builds each alternative without manufacturing measurement records" {
  for name in ["full", "partial", "unfused"] {
    let args = runtime_bundle_test_arguments("5")
    for
      item in [
        "--ingress-evaluate", name, "256", "1", "1", "128", "0", "/tmp/postprocess.cubin",
      ] {
      args.push(item)
    }
    let (selected, choice) = prepare_ingress_choice(args)
    assert_eq(
      choice,
      match name {
        "full" => @luna_fusion_strategy.FullyFused
        "partial" => @luna_fusion_strategy.ProducerSeparated
        _ => @luna_fusion_strategy.Unfused
      },
    )
    assert_eq(
      selected.ingress.module_path,
      if name == "partial" {
        "/tmp/postprocess.cubin"
      } else {
        "/tmp/ingress.cubin"
      },
    )
    args[args.length() - 7] = "fastest"
    try prepare_ingress_choice(args) catch {
      error => assert_eq(error, InvalidArguments)
    } noraise {
      _ => fail("unknown evaluation schedule accepted")
    }
  }
}

///|
test "offline comparison emits exact full partial or absent ingress family" {
  for entry in [("10", "20", "30"), ("20", "10", "30"), ("20", "30", "10")] {
    let (full, partial, unfused) = entry
    let (args, choice) = prepare_ingress_choice(
      ingress_comparison_arguments(full, partial, unfused),
    )
    let expected : @luna_fusion_strategy.FusionSchedule = if full == "10" {
      FullyFused
    } else if partial == "10" {
      ProducerSeparated
    } else {
      Unfused
    }
    assert_eq(choice, expected)
    let modules = selected_ingress_modules(
      args, choice, b"residual", b"ingress",
    )
    let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
      model_content_sha256=args.model_content,
      model_plan_sha256=args.model_plan,
      target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
      modules~,
    )
    let runtime = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
      canonical,
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    )
    match choice {
      FullyFused => {
        guard runtime.qwen_full_attention_ingress_module() is Some(module_) else {
          fail("full missing")
        }
        assert_eq(module_.operation_span(), (2, 4))
        assert_eq(args.ingress.module_path, "/tmp/ingress.cubin")
        assert_true(runtime.qwen_qknorm_rope_kv_write_module() is None)
      }
      ProducerSeparated => {
        guard runtime.qwen_qknorm_rope_kv_write_module() is Some(module_) else {
          fail("partial missing")
        }
        assert_eq(module_.operation_span(), (3, 3))
        assert_eq(args.ingress.module_path, "/tmp/postprocess.cubin")
        assert_true(runtime.qwen_full_attention_ingress_module() is None)
      }
      Unfused => {
        assert_eq(modules.length(), 1)
        assert_true(runtime.qwen_full_attention_ingress_module() is None)
        assert_true(runtime.qwen_qknorm_rope_kv_write_module() is None)
        assert_true(runtime.qwen_paged_attention_readonly_module() is None)
      }
    }
  }
  assert_eq(
    prepare_ingress_choice(runtime_bundle_test_arguments("5")).1,
    FullyFused,
  )
}

///|
test "offline ingress comparisons reject incomplete noncanonical or under-sampled input" {
  let insufficient = ingress_comparison_arguments("1", "2", "3")
  insufficient[insufficient.length() - 7] = "2"
  for
    args in [
      ingress_comparison_arguments("01", "2", "3"),
      ingress_comparison_arguments("0", "2", "3"),
      insufficient,
    ] {
    try prepare_ingress_choice(args) catch {
      error => assert_eq(error, InvalidArguments)
    } noraise {
      _ => fail("invalid comparison accepted")
    }
  }
}
