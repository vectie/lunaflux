// Offline matched-input final head and RF update. All model/update arithmetic
// is called in the installed original modules, not reimplemented in this bridge.
#include "original_h3_loading.h"
#include <cstring>

static constexpr bool valid_document_shape(int primary,int packed) {
  return primary>0&&primary<=packed;
}
static_assert(valid_document_shape(1009,1024));
static_assert(valid_document_shape(2785,2816));
static_assert(!valid_document_shape(2785,1024));

static int packed_document_rows(const std::string &base) {
  std::ifstream docs(base+"/documents.i32",std::ios::binary);
  int32_t boundaries[3];docs.read(reinterpret_cast<char*>(boundaries),sizeof(boundaries));
  if(!docs||boundaries[0]!=0||!valid_document_shape(boundaries[1],boundaries[2])) {
    PyErr_SetString(PyExc_ValueError,"literal primary/padding document geometry differs");
    return 0;
  }
  return boundaries[2];
}

static PyObject *indices(PyObject *torch,PyObject *dtype,const std::string &base,int tag,int &count) {
  std::ifstream docs(base+"/documents.i32",std::ios::binary),tags(base+"/combined.i32",std::ios::binary);
  int32_t boundaries[3];docs.read(reinterpret_cast<char*>(boundaries),sizeof(boundaries));
  if(!docs||boundaries[0]!=0||!valid_document_shape(boundaries[1],boundaries[2]))return nullptr;
  Owned values(PyList_New(0));count=0;
  for(int row=0;row<boundaries[1];++row) {
    int32_t value;tags.read(reinterpret_cast<char*>(&value),sizeof(value));
    if(!tags)return nullptr;
    if(value%3==tag) { Owned n(PyLong_FromLong(row));if(PyList_Append(values.p,n.p)<0)return nullptr;++count; }
  }
  Owned fn(PyObject_GetAttrString(torch,"tensor")),args(PyTuple_Pack(1,values.p));
  Owned kw(Py_BuildValue("{s:O,s:s}","dtype",dtype,"device","cuda"));
  return fn.p?PyObject_Call(fn.p,args.p,kw.p):nullptr;
}

#include "initial_h3_upstream.h"

static bool flow(PyObject *torch,PyObject *f32,PyObject *u8,const std::string &base,
                 const char *out,const char *label,PyObject *velocity,int rows,int width,
                 float sigma,float next,float timestep) {
  Owned x0(literal_tensor(torch,f32,(base+"/"+label+"-state.f32").c_str(),rows,width,4));
  Owned x(x0.p?PyObject_CallMethod(x0.p,"view","ii",rows,width):nullptr);
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral"));
  Owned tensor(PyObject_GetAttrString(torch,"tensor")),args(Py_BuildValue("(d)",double(timestep)));
  Owned kw(Py_BuildValue("{s:O,s:s}","dtype",f32,"device","cuda"));
  Owned t(tensor.p?PyObject_Call(tensor.p,args.p,kw.p):nullptr);
  Owned clean(module.p&&x.p&&t.p?PyObject_CallMethod(module.p,"minimax_h3_rf_v_to_x0","OOO",x.p,velocity,t.p):nullptr);
  Owned fn(module.p?PyObject_GetAttrString(module.p,"minimax_h3_euler_eta0_step"):nullptr);
  Owned operands(clean.p?PyTuple_Pack(2,x.p,clean.p):nullptr);
  Owned options(Py_BuildValue("{s:d,s:d}","sigma_curr",double(sigma),"sigma_next",double(next)));
  Owned result(fn.p&&operands.p?PyObject_Call(fn.p,operands.p,options.p):nullptr);
  return result.p&&save(result.p,u8,std::string(out)+"/"+label+"-updated.f32",int64_t(rows)*width*4);
}

static int execute(const char *root,const char *input,const char *out) {
  Owned torch(PyImport_ImportModule("torch")),safe(PyImport_ImportModule("safetensors"));
  Owned distributed(PyImport_ImportModule("vllm.distributed")),config_module(PyImport_ImportModule("vllm.config"));
  Owned config_cls(config_module.p?PyObject_GetAttrString(config_module.p,"VllmConfig"):nullptr);
  Owned config(config_cls.p?PyObject_CallNoArgs(config_cls.p):nullptr);
  Owned scope(config.p?PyObject_CallMethod(config_module.p,"set_current_vllm_config","O",config.p):nullptr);
  Owned entered(scope.p?PyObject_CallMethod(scope.p,"__enter__",nullptr):nullptr);
  if(!torch.p||!safe.p||!distributed.p||!entered.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),f32(PyObject_GetAttrString(torch.p,"float32"));
  Owned i32(PyObject_GetAttrString(torch.p,"int32")),i64(PyObject_GetAttrString(torch.p,"int64")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned init(PyObject_GetAttrString(distributed.p,"init_distributed_environment")),empty(PyTuple_New(0));
  Owned opts(Py_BuildValue("{s:i,s:i,s:i,s:s,s:s}","world_size",1,"rank",0,"local_rank",0,
    "distributed_init_method","file:///work/rendezvous","backend","gloo"));
  Owned world(init.p?PyObject_Call(init.p,empty.p,opts.p):nullptr);
  Owned parallel(PyObject_CallMethod(distributed.p,"initialize_model_parallel","ii",1,1));
  if(!grad.p||!threads.p||!dtype.p||!world.p||!parallel.p)return -1;
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer"));
  Owned metadata(json_file(std::string(root)+"/config.json"));
  Owned arch_cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTArchConfig"):nullptr);
  Owned arch(arch_cls.p&&metadata.p?PyObject_CallMethod(arch_cls.p,"from_mapping","O",metadata.p):nullptr);
  Owned cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3FinalLayer"):nullptr);
  Owned ctor(arch.p?PyTuple_Pack(2,arch.p,Py_None):nullptr);
  Owned head(cls.p&&ctor.p?PyObject_CallObject(cls.p,ctor.p):nullptr);
  Owned gpu(head.p?PyObject_CallMethod(head.p,"to","s","cuda"):nullptr);
  Owned index(json_file(std::string(root)+"/diffusion_pytorch_model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  if(!gpu.p||!map||!open.p)return -1;
  for(const auto &pair:std::vector<std::pair<const char*,const char*>>{
      {"norm","norm_out.norm"},{"video_out","proj_out"},{"audio_out","audio_proj_out"}}) {
    Owned child(PyObject_GetAttrString(gpu.p,pair.first));
    if(!child.p||!h3_parameter(child.p,open.p,map,root,"weight",(std::string(pair.second)+".weight").c_str()))return -1;
    if(std::strcmp(pair.first,"norm")&&
       !h3_parameter(child.p,open.p,map,root,"bias",(std::string(pair.second)+".bias").c_str()))return -1;
  }
  Owned adaln(PyObject_GetAttrString(gpu.p,"adaln_proj"));
  Owned linear(adaln.p?PyObject_GetAttrString(adaln.p,"linear"):nullptr);
  if(!linear.p||!h3_parameter(linear.p,open.p,map,root,"weight","norm_out.linear.weight")||
     !h3_parameter(linear.p,open.p,map,root,"bias","norm_out.linear.bias"))return -1;
  const std::string base(input);
  const int packed_rows=packed_document_rows(base);
  if(!packed_rows)return -1;
  Owned x0(literal_bf16(torch.p,bf16.p,(base+"/block-49.bf16").c_str(),packed_rows,5376));
  Owned x(x0.p?PyObject_CallMethod(x0.p,"view","ii",packed_rows,5376):nullptr);
  std::ifstream time_file(base+"/timestep.f32",std::ios::binary|std::ios::ate);
  const auto bytes=time_file.tellg();
  if(!time_file||bytes<=0||bytes%(2688*4)!=0)return -1;
  const int times=static_cast<int>(bytes/(2688*4));
  Owned t0(literal_tensor(torch.p,f32.p,(base+"/timestep.f32").c_str(),times,2688,4));
  Owned t(t0.p?PyObject_CallMethod(t0.p,"view","ii",times,2688):nullptr);
  Owned inverse0(literal_tensor(torch.p,i32.p,(base+"/inverse.i32").c_str(),packed_rows,1,4));
  Owned inverse_flat(inverse0.p?PyObject_CallMethod(inverse0.p,"view","i",packed_rows):nullptr);
  Owned inverse(inverse_flat.p?PyObject_CallMethod(inverse_flat.p,"long",nullptr):nullptr);
  Owned args(x.p?PyTuple_Pack(1,x.p):nullptr);
  Owned kwargs(t.p&&inverse.p?Py_BuildValue("{s:O,s:O}","t_emb",t.p,"inverse_indices",inverse.p):nullptr);
  Owned result(args.p&&kwargs.p?PyObject_Call(gpu.p,args.p,kwargs.p):nullptr);
  if(!result.p)return -1;
  float sigmas[6];std::ifstream sf(base+"/sigmas.f32",std::ios::binary);
  sf.read(reinterpret_cast<char*>(sigmas),sizeof(sigmas));if(!sf)return -1;
  if(!initial_producers(torch.p,module.p,arch.p,open.p,map,bf16.p,f32.p,i64.p,u8.p,
                        root,base,out,packed_rows,times,sigmas))return -1;
  for(int modality=0;modality<2;++modality) {
    const char *label=modality?"audio":"video";const int width=modality?32:96;
    int rows=0;Owned ids(indices(torch.p,i64.p,base,modality?2:0,rows));
    auto *all=PyTuple_GetItem(result.p,modality);
    Owned selected(all&&ids.p?PyObject_CallMethod(all,"index_select","iO",0,ids.p):nullptr);
    if(!selected.p||!save(selected.p,u8.p,std::string(out)+"/"+label+"-velocity.f32",int64_t(rows)*width*4))return -1;
    Owned native0(literal_tensor(torch.p,f32.p,(base+"/"+label+"-velocity.f32").c_str(),rows,width,4));
    Owned native(native0.p?PyObject_CallMethod(native0.p,"view","ii",rows,width):nullptr);
    if(!native.p||!flow(torch.p,f32.p,u8.p,base,out,label,native.p,rows,width,
                        sigmas[modality*3],sigmas[modality*3+1],sigmas[modality*3+2]))return -1;
    std::printf("original_final_head modality=%s rows=%d width=%d literal_block49=true original_flow_on_native_velocity=true\n",label,rows,width);
  }
  Owned closed(PyObject_CallMethod(scope.p,"__exit__","OOO",Py_None,Py_None,Py_None));
  return closed.p?0:-1;
}

int main(int argc,char **argv) {
  if(argc!=4)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
