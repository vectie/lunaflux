///|
test "joint cancellation options are explicit and rejected before model access" {
  let ordinary = ["exe", "run-joint", "/missing-model"]
  assert_eq(joint_cancellation_arguments(ordinary).1, 0)
  let explicit = [
    "exe", "run-joint", "/missing-model", "--cancel-request", "0", "--after-steps",
    "2",
  ]
  let (plain, count) = joint_cancellation_arguments(explicit)
  assert_eq(plain.to_owned(), ordinary)
  assert_eq(count, 2)
  for
    arguments in [
      ["exe", "run-joint", "/x", "--cancel-request", "1", "--after-steps", "2"],
      [
        "exe", "encode-text", "/x", "--cancel-request", "0", "--after-steps", "2",
      ],
      ["exe", "run-joint", "/x", "--cancel-request", "0", "--after-steps", "0"],
      ["exe", "run-joint", "/x", "--after-steps", "2"],
    ] {
    try joint_cancellation_arguments(arguments) catch {
      _ => ()
    } noraise {
      _ => fail("invalid cancellation control accepted")
    }
  }
}

///|
test "joint driver cancels between completed frames and closes before returning" {
  let stages : Array[@execution.PipelineStage] = [
    Conditioning,
    Denoising,
    VideoDecode,
    AudioDecode,
    Completed,
  ]
  let log : Array[Int] = []
  let cursor = Ref(0)
  let result = advance_joint_frames(
    () => {
      let i = cursor.val
      cursor.val += 1
      log.push(i)
      stages[i]
    },
    () => log.push(100),
    () => log.push(200),
    2,
  )
  assert_true(result.0 == Cancelled)
  assert_eq(result.1, 2)
  assert_eq(log, [0, 1, 100, 200])
  assert_eq(cursor.val, 2)
}

///|
test "joint completion wins at the last frame and no hidden limit is introduced" {
  for limit in [0, 3, 10] {
    let cursor = Ref(0)
    let cancels = Ref(0)
    let result = advance_joint_frames(
      () => {
        cursor.val += 1
        if cursor.val == 3 {
          Completed
        } else {
          Denoising
        }
      },
      () => cancels.val += 1,
      () => fail("output owner closed before publication"),
      limit,
    )
    assert_true(result.0 == Completed)
    assert_eq(result.1, 3)
    assert_eq(cancels.val, 0)
  }
}

///|
test "joint cancellation cleanup and real execution failures propagate" {
  let cancel_calls = Ref(0)
  try
    advance_joint_frames(
      () => Denoising,
      () => cancel_calls.val += 1,
      () => fail("injected close failure"),
      1,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("cancelled result escaped failed cleanup")
  }
  assert_eq(cancel_calls.val, 1)
  let failures : Array[@execution.PipelineStage] = [Failed, Closed]
  for stage in failures {
    try
      advance_joint_frames(
        () => stage,
        () => cancel_calls.val += 1,
        () => fail("driver replaced outer failure cleanup"),
        1,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("real terminal failure swallowed")
    }
  }
  assert_eq(cancel_calls.val, 1)
}

///|
test "joint driver has no inherited million-poll lifecycle ceiling" {
  let completed = Ref(0)
  let cancels = Ref(0)
  let result = advance_joint_frames(
    () => {
      completed.val += 1
      if completed.val == 1048577 {
        Completed
      } else {
        Denoising
      }
    },
    () => cancels.val += 1,
    () => fail("successful output closed"),
    0,
  )
  assert_true(result.0 == Completed)
  assert_eq(result.1, 1048577)
  assert_eq(cancels.val, 0)
}
