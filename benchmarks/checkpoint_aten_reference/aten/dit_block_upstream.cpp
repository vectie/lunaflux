// Offline original H3 block. All math stays in the installed unmodified module;
// literal serving input/timestep/position/document tensors are supplied by replay.
#include "original_tensor_io.h"

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

static bool parameter(PyObject *object, PyObject *open, PyObject *map,
                      const char *root, const char *target, const char *source,
                      const char *part=nullptr) {
  Owned destination(PyObject_GetAttrString(object,target));
  Owned value(weight(open,map,root,source));
  if(!destination.p||!value.p)return false;
  if(part) {
    Owned loader(PyObject_GetAttrString(destination.p,"weight_loader"));
    Owned args(Py_BuildValue("OOs",destination.p,value.p,part));
    Owned copied(loader.p?PyObject_CallObject(loader.p,args.p):nullptr);
    return copied.p!=nullptr;
  }
  Owned copied(PyObject_CallMethod(destination.p,"copy_","O",value.p));
  return copied.p!=nullptr;
}

static bool load_block(PyObject *block,PyObject *open,PyObject *map,const char *root) {
  Owned attention(PyObject_GetAttrString(block,"attn"));
  Owned qkv(attention.p?PyObject_GetAttrString(attention.p,"qkv_proj"):nullptr);
  if(!qkv.p)return false;
  for(const char *part:{"q","k","v"}) {
    const std::string name="transformer_blocks.0.attn.to_"+std::string(part)+".weight";
    if(!parameter(qkv.p,open,map,root,"weight",name.c_str(),part))return false;
  }
  for(const auto &pair:std::vector<std::pair<const char*,const char*>>{
      {"q_norm","attn.norm_q.weight"},{"k_norm","attn.norm_k.weight"},
      {"out_proj","attn.to_out.0.weight"}}) {
    Owned child(PyObject_GetAttrString(attention.p,pair.first));
    const std::string source="transformer_blocks.0."+std::string(pair.second);
    if(!child.p||!parameter(child.p,open,map,root,"weight",source.c_str()))return false;
  }
  for(const char *name:{"norm1","norm2"}) {
    Owned child(PyObject_GetAttrString(block,name));
    const std::string source="transformer_blocks.0."+std::string(name)+".weight";
    if(!child.p||!parameter(child.p,open,map,root,"weight",source.c_str()))return false;
  }
  Owned mlp(PyObject_GetAttrString(block,"mlp"));
  Owned fc1(mlp.p?PyObject_GetAttrString(mlp.p,"fc1"):nullptr);
  Owned fc2(mlp.p?PyObject_GetAttrString(mlp.p,"fc2"):nullptr);
  Owned target(fc1.p?PyObject_GetAttrString(fc1.p,"weight"):nullptr);
  Owned loader(target.p?PyObject_GetAttrString(target.p,"weight_loader"):nullptr);
  Owned fused(weight(open,map,root,"transformer_blocks.0.ff.net.0.proj.weight"));
  if(!loader.p||!fused.p||!fc2.p)return false;
  for(int shard=0;shard<2;++shard) {
    Owned half(PyObject_CallMethod(fused.p,"narrow","iii",0,shard*14336,14336));
    Owned args(half.p?Py_BuildValue("OOi",target.p,half.p,shard):nullptr);
    Owned copied(args.p?PyObject_CallObject(loader.p,args.p):nullptr);
    if(!copied.p)return false;
  }
  if(!parameter(fc2.p,open,map,root,"weight","transformer_blocks.0.ff.net.2.weight"))return false;
  Owned adaln(PyObject_GetAttrString(block,"adaln_proj"));
  Owned linear(adaln.p?PyObject_GetAttrString(adaln.p,"linear"):nullptr);
  return linear.p&&parameter(linear.p,open,map,root,"weight","transformer_blocks.0.adaln_proj.linear.weight")&&
    parameter(linear.p,open,map,root,"bias","transformer_blocks.0.adaln_proj.linear.bias");
}

static int execute(const char *root,const char *input,const char *out) {
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
  Owned index(json_file(std::string(root)+"/diffusion_pytorch_model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTBlock"):nullptr);
  Owned ctor(arch.p?PyTuple_Pack(2,arch.p,Py_None):nullptr);
  Owned block(cls.p&&ctor.p?PyObject_CallObject(cls.p,ctor.p):nullptr);
  Owned gpu(block.p?PyObject_CallMethod(block.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!map||!open.p||!load_block(gpu.p,open.p,map,root))return -1;
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
  Owned args(x.p?PyTuple_Pack(1,x.p):nullptr);
  Owned kw(t.p&&indices.p&&frequencies.p&&docs.p?Py_BuildValue("{s:O,s:O,s:O,s:O,s:i}",
    "t_emb",t.p,"combined_indices",indices.p,"rope_freqs",frequencies.p,
    "cu_seqlens",docs.p,"max_seqlen",1009):nullptr);
  Owned result(args.p&&kw.p?PyObject_Call(gpu.p,args.p,kw.p):nullptr);
  if(!result.p||!save(result.p,u8.p,std::string(out)+"/block-0.bf16",1024LL*5376*2))return -1;
  std::puts("original_dit_block_complete literal_inputs=true original_arithmetic=true media_quality=false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=4)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
