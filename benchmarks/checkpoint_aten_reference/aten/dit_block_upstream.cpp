// Offline original H3 block. All math stays in the installed unmodified module;
// literal serving input/timestep/position/document tensors are supplied by replay.
#include "original_h3_loading.h"
#include <cstdlib>

// The original vLLM modules require their public configuration context even
// in a one-block offline replay. This changes setup, not module arithmetic.
struct OriginalConfigScope {
  PyObject *context=nullptr;
  ~OriginalConfigScope() {
    if(!context)return;
    PyObject *type=nullptr,*value=nullptr,*trace=nullptr;
    PyErr_Fetch(&type,&value,&trace);
    Owned closed(PyObject_CallMethod(context,"__exit__","OOO",Py_None,Py_None,Py_None));
    Py_DECREF(context);
    if(type)PyErr_Restore(type,value,trace);
  }
};

static bool load_block(PyObject *block,PyObject *open,PyObject *map,const char *root,int number) {
  const std::string prefix="transformer_blocks."+std::to_string(number)+".";
  if(!h3_attention_mlp(block,open,map,root,prefix))return false;
  Owned adaln(PyObject_GetAttrString(block,"adaln_proj"));
  Owned linear(adaln.p?PyObject_GetAttrString(adaln.p,"linear"):nullptr);
  return linear.p&&h3_parameter(linear.p,open,map,root,"weight",(prefix+"adaln_proj.linear.weight").c_str())&&
    h3_parameter(linear.p,open,map,root,"bias",(prefix+"adaln_proj.linear.bias").c_str());
}

// Independently exercise the installed original checkpoint conversion, rather
// than trusting the previously converted diffusers shard set as an oracle.
static bool load_raw_block(PyObject *block,PyObject *module,PyObject *arch,
                           PyObject *open,PyObject *map,const char *root,int number) {
  Owned parameters(PyObject_CallMethod(block,"named_parameters",nullptr));
  Owned iterator(parameters.p?PyObject_GetIter(parameters.p):nullptr);
  if(!iterator.p)return false;
  while(auto *next=PyIter_Next(iterator.p)) {
    Owned entry(next);
    const char *local=PyUnicode_AsUTF8(PyTuple_GetItem(entry.p,0));
    auto *target=PyTuple_GetItem(entry.p,1);
    if(!local||!target)return false;
    const std::string name="blocks."+std::to_string(number)+"."+local;
    Owned value(weight(open,map,root,name.c_str()));
    Owned factory(PyObject_GetAttrString(module,"_prepare_h3_checkpoint_weight"));
    Owned args(value.p?Py_BuildValue("sO",name.c_str(),value.p):nullptr);
    Owned kw(Py_BuildValue("{s:O}","arch",arch));
    Owned prepared(factory.p&&args.p?PyObject_Call(factory.p,args.p,kw.p):nullptr);
    Owned copied(prepared.p?PyObject_CallMethod(target,"copy_","O",prepared.p):nullptr);
    if(!copied.p)return false;
  }
  return !PyErr_Occurred();
}

static bool audit_weights(PyObject *torch,PyObject *cls,PyObject *ctor,PyObject *raw_block,
                          PyObject *open,PyObject *raw_map,const char *raw_root,const char *canonical,const char *out) {
  Owned index(json_file(std::string(canonical)+"/diffusion_pytorch_model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned block(PyObject_CallObject(cls,ctor));
  Owned gpu(block.p?PyObject_CallMethod(block.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!map||!load_block(gpu.p,open,map,canonical,0))return false;
  Owned raw_params(PyObject_CallMethod(raw_block,"named_parameters",nullptr));
  Owned dict_cls(PyObject_GetAttrString(PyImport_AddModule("builtins"),"dict"));
  Owned raw_dict(raw_params.p?PyObject_CallFunctionObjArgs(dict_cls.p,raw_params.p,nullptr):nullptr);
  Owned params(PyObject_CallMethod(gpu.p,"named_parameters",nullptr));
  Owned iterator(params.p?PyObject_GetIter(params.p):nullptr);
  std::ofstream report(std::string(out)+"/weight-layout-with-refiners.tsv");
  report<<"parameter\texact\telements\trelative_l2\thalf_swapped_exact\n";
  if(!raw_dict.p||!iterator.p||!report)return false;
  while(auto *next=PyIter_Next(iterator.p)) {
    Owned entry(next);
    auto *name=PyTuple_GetItem(entry.p,0),*value=PyTuple_GetItem(entry.p,1);
    auto *raw=PyDict_GetItem(raw_dict.p,name);
    Owned a(PyObject_CallMethod(value,"float",nullptr));
    Owned b(raw?PyObject_CallMethod(raw,"float",nullptr):nullptr);
    Owned equal(raw?PyObject_CallMethod(value,"eq","O",raw):nullptr);
    Owned count(equal.p?PyObject_CallMethod(equal.p,"sum",nullptr):nullptr);
    Owned count_item(count.p?PyObject_CallMethod(count.p,"item",nullptr):nullptr);
    Owned extent(PyObject_CallMethod(value,"numel",nullptr));
    Owned diff(a.p&&b.p?PyNumber_Subtract(a.p,b.p):nullptr);
    Owned norm(diff.p?PyObject_CallMethod(diff.p,"norm",nullptr):nullptr);
    Owned base(b.p?PyObject_CallMethod(b.p,"norm",nullptr):nullptr);
    Owned ratio(norm.p&&base.p?PyNumber_TrueDivide(norm.p,base.p):nullptr);
    Owned item(ratio.p?PyObject_CallMethod(ratio.p,"item",nullptr):nullptr);
    if(!count_item.p||!extent.p||!item.p)return false;
    long long swapped_exact=-1;
    if(std::string(PyUnicode_AsUTF8(name))=="mlp.fc1.weight") {
      Owned shifted(PyObject_CallMethod(value,"roll","ii",14336,0));
      Owned same(shifted.p?PyObject_CallMethod(shifted.p,"eq","O",raw):nullptr);
      Owned sum(same.p?PyObject_CallMethod(same.p,"sum",nullptr):nullptr);
      Owned exact(sum.p?PyObject_CallMethod(sum.p,"item",nullptr):nullptr);
      if(!exact.p)return false;
      swapped_exact=PyLong_AsLongLong(exact.p);
    }
    report<<PyUnicode_AsUTF8(name)<<'\t'<<PyLong_AsLongLong(count_item.p)<<'\t'
          <<PyLong_AsLongLong(extent.p)<<'\t'<<PyFloat_AsDouble(item.p)<<'\t'<<swapped_exact<<'\n';
    report.flush();
  }
  for(int layer=0;layer<2;++layer) {
    const std::string name="token_refiner.refiner_blocks."+std::to_string(layer)+".ff.net.0.proj.weight";
    const std::string raw_name="token_refiner.blocks."+std::to_string(layer)+".mlp.fc1.weight";
    Owned a(weight(open,map,canonical,name.c_str()));
    Owned b(weight(open,raw_map,raw_root,raw_name.c_str()));
    Owned shifted(a.p?PyObject_CallMethod(a.p,"roll","ii",14336,0):nullptr);
    Owned same(a.p&&b.p?PyObject_CallMethod(a.p,"eq","O",b.p):nullptr);
    Owned swapped(shifted.p&&b.p?PyObject_CallMethod(shifted.p,"eq","O",b.p):nullptr);
    Owned sum(same.p?PyObject_CallMethod(same.p,"sum",nullptr):nullptr);
    Owned swapped_sum(swapped.p?PyObject_CallMethod(swapped.p,"sum",nullptr):nullptr);
    Owned exact(sum.p?PyObject_CallMethod(sum.p,"item",nullptr):nullptr);
    Owned swapped_exact(swapped_sum.p?PyObject_CallMethod(swapped_sum.p,"item",nullptr):nullptr);
    Owned extent(a.p?PyObject_CallMethod(a.p,"numel",nullptr):nullptr);
    if(!exact.p||!swapped_exact.p||!extent.p)return false;
    report<<name<<'\t'<<PyLong_AsLongLong(exact.p)<<'\t'<<PyLong_AsLongLong(extent.p)
          <<"\tNA\t"<<PyLong_AsLongLong(swapped_exact.p)<<'\n';
    report.flush();
  }
  (void)torch;
  return !PyErr_Occurred();
}

static int execute(const char *root,const char *input,const char *out,int blocks,bool raw,const char *canonical) {
  Owned torch(PyImport_ImportModule("torch")),safe(PyImport_ImportModule("safetensors"));
  Owned distributed(PyImport_ImportModule("vllm.distributed"));
  if(!torch.p||!safe.p||!distributed.p)return -1;
  Owned configuration(PyImport_ImportModule("vllm.config"));
  Owned vcfg_cls(configuration.p?PyObject_GetAttrString(configuration.p,"VllmConfig"):nullptr);
  Owned vcfg(vcfg_cls.p?PyObject_CallNoArgs(vcfg_cls.p):nullptr);
  OriginalConfigScope scope;
  scope.context=vcfg.p?PyObject_CallMethod(configuration.p,"set_current_vllm_config","O",vcfg.p):nullptr;
  Owned entered(scope.context?PyObject_CallMethod(scope.context,"__enter__",nullptr):nullptr);
  if(!entered.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),f32(PyObject_GetAttrString(torch.p,"float32"));
  Owned i32(PyObject_GetAttrString(torch.p,"int32")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned init(PyObject_GetAttrString(distributed.p,"init_distributed_environment"));
  Owned empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i,s:i,s:s,s:s}","world_size",1,"rank",0,"local_rank",0,
    "distributed_init_method","file:///work/rendezvous","backend","gloo"));
  Owned world(init.p?PyObject_Call(init.p,empty.p,options.p):nullptr);
  Owned parallel(PyObject_CallMethod(distributed.p,"initialize_model_parallel","ii",1,1));
  if(!grad.p||!threads.p||!dtype.p||!world.p||!parallel.p)return -1;
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer"));
  Owned config(json_file(std::string(root)+"/config.json"));
  Owned arch_cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTArchConfig"):nullptr);
  Owned arch(arch_cls.p&&config.p?PyObject_CallMethod(arch_cls.p,"from_mapping","O",config.p):nullptr);
  Owned index(json_file(std::string(root)+(raw?"/model.safetensors.index.json":"/diffusion_pytorch_model.safetensors.index.json")));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTBlock"):nullptr);
  Owned ctor(arch.p?PyTuple_Pack(2,arch.p,Py_None):nullptr);
  Owned block(cls.p&&ctor.p?PyObject_CallObject(cls.p,ctor.p):nullptr);
  Owned gpu(block.p?PyObject_CallMethod(block.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!map||!open.p)return -1;
  if(canonical) {
    if(!load_raw_block(gpu.p,module.p,arch.p,open.p,map,root,0))return -1;
    return audit_weights(torch.p,cls.p,ctor.p,gpu.p,open.p,map,root,canonical,out)?0:-1;
  }
  const std::string base(input);
  Owned x0(literal_bf16(torch.p,bf16.p,(base+"/packed-input.bf16").c_str(),1024,5376));
  Owned x(x0.p?PyObject_CallMethod(x0.p,"view","ii",1024,5376):nullptr);
  // Use the captured unique-time table, never infer it from modality counts.
  // Distilled FL2VA can have one unique time shared by all packed rows.
  std::ifstream time_file(base+"/timestep.f32",std::ios::binary|std::ios::ate);
  const auto time_bytes=time_file.tellg();
  if(!time_file||time_bytes<=0||time_bytes%(2688*4)!=0) {
    PyErr_SetString(PyExc_ValueError,"captured unique-time extent differs");return -1;
  }
  const int times=static_cast<int>(time_bytes/(2688*4));
  Owned t0(literal_tensor(torch.p,f32.p,(base+"/timestep.f32").c_str(),times,2688,4));
  Owned t(t0.p?PyObject_CallMethod(t0.p,"view","ii",times,2688):nullptr);
  Owned p(literal_tensor(torch.p,f32.p,(base+"/positions.f32").c_str(),1024,3,4));
  Owned indices0(literal_tensor(torch.p,i32.p,(base+"/combined.i32").c_str(),1024,1,4));
  Owned flat_indices(indices0.p?PyObject_CallMethod(indices0.p,"view","i",1024):nullptr);
  Owned indices(flat_indices.p?PyObject_CallMethod(flat_indices.p,"long",nullptr):nullptr);
  Owned docs0(literal_tensor(torch.p,i32.p,(base+"/documents.i32").c_str(),3,1,4));
  Owned docs(docs0.p?PyObject_CallMethod(docs0.p,"view","i",3):nullptr);
  Owned rope_cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3Rope"):nullptr);
  Owned rope(rope_cls.p?PyObject_CallFunction(rope_cls.p,"i",16):nullptr);
  Owned rope_gpu(rope.p?PyObject_CallMethod(rope.p,"to","s","cuda"):nullptr);
  Owned inverse(rope_gpu.p?PyObject_GetAttrString(rope_gpu.p,"inv_freq"):nullptr);
  // The converted diffusers shard set omits this buffer. Use the exact F32
  // table already extracted from FL2VA and supplied to native serving, not a
  // recomputed frequency table or a different checkpoint default.
  Owned source0(literal_tensor(torch.p,f32.p,(std::string(root)+"/rope.inv_freq.f32").c_str(),1,16,4));
  Owned source(source0.p?PyObject_CallMethod(source0.p,"view","i",16):nullptr);
  Owned copied(inverse.p&&source.p?PyObject_CallMethod(inverse.p,"copy_","O",source.p):nullptr);
  Owned positions(p.p?PyTuple_Pack(1,p.p):nullptr);
  Owned frequencies(copied.p&&positions.p?PyObject_CallObject(rope_gpu.p,positions.p):nullptr);
  Owned kw(t.p&&indices.p&&frequencies.p&&docs.p?Py_BuildValue("{s:O,s:O,s:O,s:O,s:i}",
    "t_emb",t.p,"combined_indices",indices.p,"rope_freqs",frequencies.p,
    "cu_seqlens",docs.p,"max_seqlen",1009):nullptr);
  if(!x.p||!kw.p)return -1;
  Owned propagated(Py_NewRef(x.p));
  for(int number=0;number<blocks;++number) {
    if(raw?!load_raw_block(gpu.p,module.p,arch.p,open.p,map,root,number):
           !load_block(gpu.p,open.p,map,root,number))return -1;
    Owned previous(number?literal_bf16(torch.p,bf16.p,
      (base+"/block-"+std::to_string(number-1)+".bf16").c_str(),1024,5376):Py_NewRef(x.p));
    Owned matched(previous.p?PyObject_CallMethod(previous.p,"view","ii",1024,5376):nullptr);
    Owned args(matched.p?PyTuple_Pack(1,matched.p):nullptr);
    Owned result(args.p?PyObject_Call(gpu.p,args.p,kw.p):nullptr);
    const std::string label=std::to_string(number)+".bf16";
    if(!result.p||!save(result.p,u8.p,std::string(out)+"/"+
        (blocks==1?"block-":"matched-block-")+label,1024LL*5376*2))return -1;
    if(blocks>1) {
      Owned pargs(PyTuple_Pack(1,propagated.p));
      Owned next(pargs.p?PyObject_Call(gpu.p,pargs.p,kw.p):nullptr);
      if(!next.p||!save(next.p,u8.p,std::string(out)+"/propagated-block-"+label,1024LL*5376*2))return -1;
      Py_SETREF(propagated.p,Py_NewRef(next.p));
    }
    std::printf("original_dit_block=%d literal_predecessor=true propagated_predecessor=%s\n",number,blocks>1?"true":"false");
    std::fflush(stdout);
  }
  std::puts("original_dit_blocks_complete literal_inputs=true original_arithmetic=true media_quality=false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc<4||argc>7)return 2;
  const int blocks=argc>=5?std::atoi(argv[4]):1;
  if(blocks<1||blocks>50)return 2;
  const bool raw=argc>=6&&std::string(argv[5])=="raw";
  if(argc>=6&&!raw)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],blocks,raw,argc==7?argv[6]:nullptr);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
