///|
fn finish_source(program : @physical.FullVocabularySamplingProgram) -> String {
  let out = StringBuilder()
  for step in program.final_steps() {
    out.write_string(
      match step {
        OrderedSoftmax => "double total=0.0;for(int i=0;i<k;++i) total+=w[i];\n"
        RequireFiniteMass =>
          "if(!(total>0.0) || !isfinite(total)){results[2*r]=-1;results[2*r+1]=-3;return;}\n"
        MinimalProbabilityPrefix =>
          "int retained=k;if(p.flags&2){double threshold=p.top_p*total,cumulative=0.0;retained=0;while(retained<k && cumulative<threshold){cumulative+=w[retained];++retained;}retained=max(retained,1);}\n"
        SumRetainedMass =>
          "double mass=0.0;for(int i=0;i<retained;++i)mass+=w[i];\n"
        CounterDraw =>
          "double unit=(double)(counter(p.seed,p.sample_index)>>11)*1.1102230246251565e-16;double target=unit*mass;\n"
        OrderedInverseCdf =>
          "double cumulative=0.0;int chosen=retained-1;for(int i=0;i<retained;++i){cumulative+=w[i];if(target<cumulative){chosen=i;break;}}\n"
        PublishSample => "results[2*r]=c[chosen].token;results[2*r+1]=-1;\n"
      },
    )
  }
  out.to_string()
}
