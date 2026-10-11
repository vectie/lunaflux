#pragma once
// Offline original input producers on the same captured state and schedule.
// `indices` and checkpoint I/O are supplied by the final-head bridge.
static bool initial_producers(PyObject *torch,PyObject *module,PyObject *arch,
                              PyObject *open,PyObject *map,PyObject *bf16,
                              PyObject *f32,PyObject *i64,PyObject *u8,
                              const char *root,const std::string &base,
                              const char *out,int times,const float *sigmas) {
  Owned packed0(literal_bf16(torch,bf16,(base+"/packed-input.bf16").c_str(),1024,5376));
  Owned packed(packed0.p?PyObject_CallMethod(packed0.p,"view","ii",1024,5376):nullptr);
  Owned cls(PyObject_GetAttrString(module,"ColumnParallelLinear"));
  if(!packed.p||!cls.p)return false;
  for(int modality=0;modality<2;++modality) {
    const char *label=modality?"audio":"video";
    const char *prefix=modality?"audio_proj_in":"proj_in";
    const int width=modality?32:96;
    int rows=0;Owned ids(indices(torch,i64,base,modality?2:0,rows));
    Owned dims(Py_BuildValue("(ii)",width,5376));
    Owned kw(Py_BuildValue("{s:O,s:O,s:O,s:O}","bias",Py_True,
      "gather_output",Py_True,"params_dtype",f32,"quant_config",Py_None));
    Owned projection(PyObject_Call(cls.p,dims.p,kw.p));
    Owned gpu(projection.p?PyObject_CallMethod(projection.p,"to","s","cuda"):nullptr);
    if(!gpu.p||!h3_parameter(gpu.p,open,map,root,"weight",(std::string(prefix)+".weight").c_str())||
       !h3_parameter(gpu.p,open,map,root,"bias",(std::string(prefix)+".bias").c_str()))return false;
    Owned state0(literal_tensor(torch,f32,(base+"/"+label+"-state.f32").c_str(),rows,width,4));
    Owned state(state0.p?PyObject_CallMethod(state0.p,"view","ii",rows,width):nullptr);
    Owned args(state.p?PyTuple_Pack(1,state.p):nullptr);
    Owned pair(args.p?PyObject_CallObject(gpu.p,args.p):nullptr);
    auto *value=pair.p?PyTuple_GetItem(pair.p,0):nullptr;
    Owned converted(value?PyObject_CallMethod(value,"to","O",bf16):nullptr);
    Owned native(ids.p?PyObject_CallMethod(packed.p,"index_select","iO",0,ids.p):nullptr);
    if(!converted.p||!native.p||
       !save(converted.p,u8,std::string(out)+"/"+label+"-embedded.bf16",int64_t(rows)*5376*2)||
       !save(native.p,u8,std::string(out)+"/native-"+label+"-embedded.bf16",int64_t(rows)*5376*2))return false;
  }
  // Recover literal unique-time ownership, including the separate padding doc.
  // No sorted-index convention or assumed order is substituted.
  std::vector<float> values(times);std::vector<bool> seen(times,false);
  std::ifstream inverse(base+"/inverse.i32",std::ios::binary),tags(base+"/combined.i32",std::ios::binary);
  for(int row=0;row<1024;++row) {
    int32_t index,tag;inverse.read(reinterpret_cast<char*>(&index),4);tags.read(reinterpret_cast<char*>(&tag),4);
    if(!inverse||!tags||index<0||index>=times||tag<0)return false;
    const float value=tag%3==0?sigmas[2]:tag%3==2?sigmas[5]:1.0f;
    if(seen[index]&&values[index]!=value)return false;
    values[index]=value;seen[index]=true;
  }
  Owned sequence(PyList_New(times));
  for(int i=0;i<times;++i) {
    if(!seen[i])return false;
    PyList_SET_ITEM(sequence.p,i,PyFloat_FromDouble(values[i]));
  }
  Owned tensor(PyObject_GetAttrString(torch,"tensor")),targs(PyTuple_Pack(1,sequence.p));
  Owned tkw(Py_BuildValue("{s:O,s:s}","dtype",f32,"device","cuda"));
  Owned time(tensor.p?PyObject_Call(tensor.p,targs.p,tkw.p):nullptr);
  Owned time_cls(PyObject_GetAttrString(module,"MiniMaxH3TimeEmbedder"));
  Owned ctor(PyTuple_Pack(2,arch,Py_None));
  Owned embedder(time_cls.p?PyObject_CallObject(time_cls.p,ctor.p):nullptr);
  Owned gpu(embedder.p?PyObject_CallMethod(embedder.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!time.p)return false;
  for(const auto &pair:std::vector<std::pair<const char*,const char*>>{
      {"proj_in","time_embedder.linear_1"},{"proj_out","time_embedder.linear_2"}}) {
    Owned child(PyObject_GetAttrString(gpu.p,pair.first));
    if(!child.p||!h3_parameter(child.p,open,map,root,"weight",(std::string(pair.second)+".weight").c_str())||
       !h3_parameter(child.p,open,map,root,"bias",(std::string(pair.second)+".bias").c_str()))return false;
  }
  Owned args(PyTuple_Pack(1,time.p)),result(PyObject_CallObject(gpu.p,args.p));
  return result.p&&save(result.p,u8,std::string(out)+"/timestep.f32",int64_t(times)*2688*4);
}
