///|
fn chat_policy() -> @auth.LunaApiAuthPolicy raise {
  guard @env.get_env_var("LUNAFLUX_API_KEY") is Some(key) && !key.is_empty() else {
    fail("serve requires LUNAFLUX_API_KEY for the existing HTTP bearer policy")
  }
  let secret = @utf8.encode(key).to_fixedarray()
  @auth.LunaApiAuthPolicy::new(
    secret,
    source_offset=0,
    length=secret.length(),
    maximum_credential_bytes=4096,
  )
}

///|
fn chat_timeout() -> Int32 raise {
  let timeout = match @env.get_env_var("LUNAFLUX_REQUEST_TIMEOUT_MS") {
    Some(value) => @string.parse_int(value)
    None => 30000
  }
  guard timeout > 0 && timeout <= 300000 else { fail("invalid chat timeout") }
  timeout
}
