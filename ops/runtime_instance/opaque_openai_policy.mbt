///|
fn runtime_fixed_utf8(text : String) -> FixedArray[Byte] {
  let encoded = @utf8.encode(text)
  FixedArray::makei(encoded.length(), index => encoded[index])
}

///|
fn build_opaque_openai_policy(
  instance : AdmittedRuntimeInstance,
  auth_policy : @api_auth.LunaApiAuthPolicy,
) -> RuntimeOpenAiServerPolicy raise RuntimeInstanceError {
  guard instance.openai_service_policy() is Some(claims) else {
    raise fail(OpenAiProduction)
  }
  if claims.maximum_credential_bytes() <= 0 ||
    claims.maximum_credential_bytes() > 4096 {
    raise fail(OpenAiProduction)
  }
  let inference = instance.inference_limits()
  let template = @openai_compat.LunaOpenAIChatTemplate::new(
    inference,
    runtime_fixed_utf8(claims.system_prefix()),
    runtime_fixed_utf8(claims.system_suffix()),
    runtime_fixed_utf8(claims.user_prefix()),
    runtime_fixed_utf8(claims.user_suffix()),
    runtime_fixed_utf8(claims.assistant_prefix()),
    runtime_fixed_utf8(claims.assistant_suffix()),
    runtime_fixed_utf8(claims.assistant_cue()),
    max_rendered_prompt_bytes=claims.max_rendered_prompt_bytes(),
  ) catch {
    _ => raise fail(OpenAiProduction)
  }
  new_openai_server_policy(
    @http1.LunaHttp1Limits::new(
      max_head_bytes=claims.max_head_bytes(),
      max_body_bytes=claims.max_body_bytes(),
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    @http1.LunaHttp1StepBudget::new(claims.http_step_work_units()) catch {
      _ => raise fail(OpenAiProduction)
    },
    @http1.LunaHttp1ResponseStepBudget::new(claims.response_step_work_units()) catch {
      _ => raise fail(OpenAiProduction)
    },
    auth_policy,
    @openai_compat.LunaOpenAICompatStepBudget::new(
      claims.inbound_step_work_units(),
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    @openai_compat.LunaOpenAICompatStepBudget::new(
      claims.outbound_step_work_units(),
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    @openai_compat.LunaOpenAIInboundLimits::new(
      max_messages=claims.max_messages(),
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    template,
    runtime_fixed_utf8(claims.model_alias()),
    runtime_fixed_utf8(claims.response_id_prefix()),
    @inference.CacheScope::from_ascii(
      @utf8.encode(claims.cache_scope_ascii()),
      inference,
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    claims.max_new_tokens(),
    claims.context_ceiling(),
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(claims.sampling_seed()) catch {
      _ => raise fail(OpenAiProduction)
    },
    @inference.DeadlineBudget::from_milliseconds(
      claims.deadline_milliseconds(),
      inference,
    ) catch {
      _ => raise fail(OpenAiProduction)
    },
    OpenAiProduction,
  )
}

///|
fn[T] close_failed_opaque_openai_policy(
  policy : @api_auth.LunaApiAuthPolicy,
  error : RuntimeInstanceError,
) -> T raise RuntimeInstanceError {
  policy.close()
  raise error
}

///|
fn select_opaque_startup_binding(
  instance : AdmittedRuntimeInstance,
  credential : @inference_credential.InheritedInferenceAuthPolicy?,
) -> RuntimeOpaqueStartupBinding raise RuntimeInstanceError {
  match instance.external_protocol().mode() {
    NativeFramedV1 => {
      if credential is Some(inherited) {
        inherited.close()
        raise fail(InheritedCredentialBinding)
      }
      if instance.openai_service_policy() is Some(_) {
        raise fail(InheritedCredentialBinding)
      }
      RuntimeOpaqueNativeBinding
    }
    OpenAiResponsesV1 => {
      guard credential is Some(inherited) else {
        raise fail(InheritedCredentialBinding)
      }
      guard instance.openai_service_policy() is Some(claims) else {
        inherited.close()
        raise fail(OpenAiProduction)
      }
      let policy = inherited.take_for_policy(claims.maximum_credential_bytes()) catch {
        _ => {
          inherited.close()
          raise fail(InheritedCredentialBinding)
        }
      }
      let built = build_opaque_openai_policy(instance, policy) catch {
        error => close_failed_opaque_openai_policy(policy, error)
      }
      RuntimeOpaqueOpenAiBinding(built)
    }
  }
}

///|
fn RuntimeOpaqueStartupBinding::close_authentication(
  self : RuntimeOpaqueStartupBinding,
) -> Unit {
  match self {
    RuntimeOpaqueNativeBinding => ()
    RuntimeOpaqueOpenAiBinding(policy) => policy.auth_policy.close()
  }
}
