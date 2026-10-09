///|
fn read_u32_le(bytes : Bytes, offset : Int) -> UInt {
  bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)
}

///|
test "production preparation matches independent H3 schedules across step counts" {
  for variant in [@minimax_h3_spec.MiniMaxH3Variant::Fl2va, Ref2va] {
    let model = @minimax_h3.build(
      @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
        @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
          variant~,
          checkpoint_format=DiffusersV1,
        ),
        @spec.ContentDigest::from_sha256(String::make(64, '7')),
      ),
    )
    for count in [5, 10, 20, 50] {
      let request = model
        .shapes()
        .resolve_request(
          height=768,
          width=1344,
          requested_frames=120,
          inference_steps=count,
        )
      let prepared = @joint_diffusion_loop.PreparedPlan::new(
        model,
        request,
        max_latent_elements=1000000000L,
      )
      let video = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
        model,
        request,
        VideoLatent,
      )
      let audio = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
        model,
        request,
        AudioLatent,
      )
      assert_eq(prepared.model_digest(), model.plan_digest())
      assert_eq(prepared.request(), request)
      assert_eq(prepared.steps().length(), count)
      let table = prepared.flow_table()
      let storage = @joint_diffusion_loop.LatentStoragePlan::new(
        prepared,
        max_bytes=1000000000L,
      )
      assert_true(storage.matches_prepared(prepared))
      let buffers : Array[@joint_diffusion_loop.LatentBuffer] = [
        VideoState,
        AudioState,
        VideoVelocity,
        AudioVelocity,
        VideoBoundary,
        AudioBoundary,
        FlowTableBuffer,
      ]
      let mut end = 0L
      for index, buffer in buffers {
        let region = storage.region(buffer)
        assert_eq(region.offset % 256L, 0L)
        assert_true(region.offset >= end)
        assert_true(region.offset - end < 256L)
        let expected = if index == 6 {
          table.byte_count()
        } else if index % 2 == 0 {
          prepared.video_elements() * 4L
        } else {
          prepared.audio_elements() * 4L
        }
        assert_eq(region.byte_count, expected)
        end = region.offset + region.byte_count
      }
      assert_eq(storage.byte_count(), end)
      assert_eq(storage.flow_table().bytes(), table.bytes())
      let video_input = storage.boundary(PackVideoInput)
      let video_output = storage.boundary(UnpackVideoOutput)
      let audio_output = storage.boundary(UnpackAudioOutput)
      assert_eq(
        storage.region(video_input.input).offset,
        storage.region(VideoBoundary).offset,
      )
      assert_eq(
        storage.region(video_input.output).offset,
        storage.region(VideoState).offset,
      )
      assert_eq(
        storage.region(video_output.input).offset,
        storage.region(VideoState).offset,
      )
      assert_eq(
        storage.region(video_output.output).offset,
        storage.region(VideoBoundary).offset,
      )
      assert_eq(
        storage.region(audio_output.input).offset,
        storage.region(AudioState).offset,
      )
      assert_eq(
        storage.region(audio_output.output).offset,
        storage.region(AudioBoundary).offset,
      )
      for transfer in [video_input, video_output, audio_output] {
        let input = storage.region(transfer.input)
        let output = storage.region(transfer.output)
        assert_eq(input.byte_count, output.byte_count)
        assert_eq(input.byte_count, transfer.permutation.element_count() * 4L)
        assert_true(
          input.offset + input.byte_count <= output.offset ||
          output.offset + output.byte_count <= input.offset,
        )
      }
      for raw_index in [0L, 1L, 84L, 2016L, prepared.video_elements() - 1L] {
        let packed_index = video_output.permutation.source_index(raw_index)
        assert_eq(video_input.permutation.source_index(packed_index), raw_index)
      }
      assert_eq(audio_output.permutation.source_index(1L), 32L)
      assert_eq(audio_output.permutation.source_index(207L), 1L)
      assert_eq(audio_output.permutation.source_index(32L * 207L), 32L * 207L)
      ignore(
        @joint_diffusion_loop.LatentStoragePlan::new(prepared, max_bytes=end),
      )
      for limit in [0L, end - 1L] {
        try
          @joint_diffusion_loop.LatentStoragePlan::new(
            prepared,
            max_bytes=limit,
          )
        catch {
          @joint_diffusion_loop.StoragePlanError::ByteLimitExceeded => ()
        } noraise {
          _ => fail("undersized latent arena accepted")
        }
      }
      let payload = table.bytes()
      let projections = storage.projections(model, max_bytes=1000000000L)
      let video_projection = projections.shape(VideoLatent)
      let audio_projection = projections.shape(AudioLatent)
      assert_eq(video_projection.input_width, 96)
      assert_eq(audio_projection.input_width, 32)
      assert_eq(video_projection.rows, prepared.video_elements() / 96L)
      assert_eq(audio_projection.rows, 414L)
      assert_eq(video_projection.output.offset, 0L)
      assert_eq(audio_projection.output.offset % 256L, 0L)
      assert_true(
        audio_projection.output.offset >= video_projection.output.byte_count,
      )
      assert_true(
        audio_projection.output.offset - video_projection.output.byte_count <
        256L,
      )
      for projected in [video_projection, audio_projection] {
        assert_eq(projected.output_width, model.denoiser().hidden_size())
        assert_eq(
          projected.weight_bytes,
          projected.input_width.to_int64() *
          projected.output_width.to_int64() *
          4L,
        )
        assert_eq(projected.bias_bytes, projected.output_width.to_int64() * 4L)
        assert_eq(
          projected.output.byte_count,
          projected.rows * projected.output_width.to_int64() * 2L,
        )
      }
      assert_eq(
        projections.byte_count(),
        audio_projection.output.offset + audio_projection.output.byte_count,
      )
      ignore(storage.projections(model, max_bytes=projections.byte_count()))
      for budget in [0L, projections.byte_count() - 1L] {
        try storage.projections(model, max_bytes=budget) catch {
          @joint_diffusion_loop.ProjectionPlanError::ByteLimitExceeded => ()
          @joint_diffusion_loop.ProjectionPlanError::ModelMismatch =>
            fail("unexpected model mismatch")
        } noraise {
          _ => fail("undersized projection arena admitted")
        }
      }
      let other_model = @minimax_h3.build(
        @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
          @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
            variant~,
            checkpoint_format=DiffusersV1,
          ),
          @spec.ContentDigest::from_sha256(String::make(64, '8')),
        ),
      )
      let other_prepared = @joint_diffusion_loop.PreparedPlan::new(
        other_model,
        request,
        max_latent_elements=1000000000L,
      )
      assert_true(!storage.matches_prepared(other_prepared))
      try storage.projections(other_model, max_bytes=1000000000L) catch {
        @joint_diffusion_loop.ProjectionPlanError::ModelMismatch => ()
        @joint_diffusion_loop.ProjectionPlanError::ByteLimitExceeded =>
          fail("unexpected projection budget error")
      } noraise {
        _ => fail("foreign projection model admitted")
      }
      for seed in [42L, -1L] {
        let noise = storage.noise(seed~, max_chunk_elements=13)
        let reference = @joint_diffusion_reference.CounterBasedJointLatentRng::new(
          model,
          request,
          seed~,
          conditioning_draw_limit=0L,
        )
        for
          (latent, stream) in [
            (
              @joint_diffusion_plan.JointDiffusionLatentKind::VideoLatent,
              @joint_diffusion_reference.JointLatentRandomStream::VideoLatentStream,
            ),
            (AudioLatent, AudioLatentStream),
          ] {
          let size = noise.element_count(latent)
          assert_eq(size, reference.draw_limit(stream))
          for start in [0L, 17L, size - 13L] {
            let chunk = noise.chunk(latent, start~, count=13)
            assert_eq(chunk.length(), 52)
            for index in 0..<13 {
              let expected = Float::from_double(
                reference.normal(stream, start + index.to_int64()),
              )
              assert_eq(
                read_u32_le(chunk, index * 4),
                expected.reinterpret_as_uint(),
              )
              let single = noise.chunk(
                latent,
                start=start + index.to_int64(),
                count=1,
              )
              assert_eq(read_u32_le(single, 0), read_u32_le(chunk, index * 4))
            }
          }
          for
            (start, count) in [
              (-1L, 1),
              (0L, 0),
              (0L, 14),
              (size, 1),
              (size - 1L, 2),
            ] {
            try noise.chunk(latent, start~, count~) catch {
              @joint_diffusion_loop.NoiseError::InvalidRange => ()
              @joint_diffusion_loop.NoiseError::InvalidChunkLimit =>
                fail("unexpected chunk limit error")
            } noraise {
              _ => fail("invalid noise range accepted")
            }
          }
        }
      }
      for max_chunk_elements in [0, -1, 536870912] {
        try storage.noise(seed=42L, max_chunk_elements~) catch {
          @joint_diffusion_loop.NoiseError::InvalidChunkLimit => ()
          @joint_diffusion_loop.NoiseError::InvalidRange =>
            fail("unexpected range error")
        } noraise {
          _ => fail("invalid noise chunk ceiling accepted")
        }
      }
      assert_eq(table.byte_count(), count.to_int64() * 32L)
      // First record starts with IEEE F32 1.0, little endian, not host endian.
      assert_eq(payload[0], b'\x00')
      assert_eq(payload[1], b'\x00')
      assert_eq(payload[2], b'\x80')
      assert_eq(payload[3], b'\x3f')
      for index, step in prepared.steps() {
        for
          (latent, values, time) in [
            (
              @joint_diffusion_plan.JointDiffusionLatentKind::VideoLatent,
              step.video_coefficients(),
              step.video_timestep,
            ),
            (AudioLatent, step.audio_coefficients(), step.audio_timestep),
          ] {
          let offset = table.coefficient_offset(index, latent).to_int()
          assert_eq(offset % 16, 0)
          assert_eq(
            read_u32_le(payload, offset),
            values.sigma_t.reinterpret_as_uint(),
          )
          assert_eq(
            read_u32_le(payload, offset + 4),
            values.ratio.reinterpret_as_uint(),
          )
          assert_eq(
            read_u32_le(payload, offset + 8),
            values.complement.reinterpret_as_uint(),
          )
          assert_eq(
            read_u32_le(payload, offset + 12),
            Float::from_double(time).reinterpret_as_uint(),
          )
        }
      }
      assert_eq(read_u32_le(payload, (count - 1) * 32 + 4), 0U)
      assert_eq(read_u32_le(payload, (count - 1) * 32 + 8), 0x3f800000U)
      for invalid_index in [-1, count] {
        try table.coefficient_offset(invalid_index, VideoLatent) catch {
          @joint_diffusion_loop.FlowTableError::StepOutOfBounds => ()
        } noraise {
          _ => fail("out-of-range coefficient record accepted")
        }
      }
      let pack = prepared.video_pack()
      let audio_unpack = prepared.audio_unpack()
      assert_eq(pack.element_count(), prepared.video_elements())
      assert_eq(audio_unpack.element_count(), prepared.audio_elements())
      // First video patch spans 2x2 pixels within each raw channel plane.
      assert_eq(pack.source_index(0L), 0L)
      assert_eq(pack.source_index(1L), 1L)
      assert_eq(pack.source_index(2L), 84L)
      assert_eq(
        pack.source_index(4L),
        request.video_latent_frames().to_int64() * 48L * 84L,
      )
      assert_eq(pack.inverse().source_index(84L), 2L)
      // Output [stream,latent-channel,time] reads [stream,time,latent-channel].
      assert_eq(audio_unpack.source_index(1L), 32L)
      assert_eq(audio_unpack.source_index(207L), 1L)
      assert_eq(audio_unpack.source_index(32L * 207L), 32L * 207L)
      assert_eq(
        prepared.video_elements(),
        24L * request.video_latent_frames().to_int64() * 48L * 84L,
      )
      assert_eq(
        prepared.audio_elements(),
        model.shapes().audio_latent_channels().to_int64() *
        request.audio_token_rows().to_int64(),
      )
      for i, step in prepared.steps() {
        assert_eq(step.video_timestep, 1.0 - video.sigma(i).unwrap())
        assert_eq(step.audio_timestep, 1.0 - audio.sigma(i).unwrap())
        assert_eq(step.video_sigma, video.sigma(i).unwrap())
        assert_eq(step.audio_sigma, audio.sigma(i).unwrap())
        assert_eq(step.video_next_sigma, video.sigma(i + 1).unwrap())
        assert_eq(step.audio_next_sigma, audio.sigma(i + 1).unwrap())
      }
      let exact = prepared.video_elements() + prepared.audio_elements()
      ignore(
        @joint_diffusion_loop.PreparedPlan::new(
          model,
          request,
          max_latent_elements=exact,
        ),
      )
      for ceiling in [0L, exact - 1L] {
        try
          @joint_diffusion_loop.PreparedPlan::new(
            model,
            request,
            max_latent_elements=ceiling,
          )
        catch {
          @joint_diffusion_loop.PreparationError::ElementLimitExceeded => ()
          _ => fail("wrong capacity error")
        } noraise {
          _ => fail("insufficient capacity admitted")
        }
      }
      let first = prepared.start()
      let second = prepared.start()
      first.cancel()
      assert_eq(first.progress(), Cancelled)
      assert_eq(second.progress(), Running)
      let numeric = backend()
      for _ in 0..<count {
        ignore(second.advance(numeric))
      }
      assert_eq(second.progress(), Completed)
      assert_eq(second.completed_steps(), count)
      assert_eq(numeric.calls, count)
    }
  }
}
