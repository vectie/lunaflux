// Offline full original trajectory. The installed original model, packing and
// denoise loop own all arithmetic. This bridge only supplies checkpoint layout,
// literal native operands and observations; it is not a production dependency.
#include "original_h3_loading.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

static bool linear(PyObject *parent,const char *child,PyObject *open,PyObject *map,
                   const char *root,const std::string &prefix,bool bias=true) {
  Owned module(PyObject_GetAttrString(parent,child));
  return module.p&&h3_parameter(module.p,open,map,root,"weight",(prefix+".weight").c_str())&&
    (!bias||h3_parameter(module.p,open,map,root,"bias",(prefix+".bias").c_str()));
}

static bool load_model(PyObject *model,PyObject *open,PyObject *map,const char *root) {
  for(const auto &pair:std::vector<std::pair<const char*,const char*>>{
      {"video_patch_proj","proj_in"},{"audio_patch_proj","audio_proj_in"},
      {"condition_proj","context_embedder"}})
    if(!linear(model,pair.first,open,map,root,pair.second))return false;
  Owned time(PyObject_GetAttrString(model,"time_embedder"));
  if(!time.p||!linear(time.p,"proj_in",open,map,root,"time_embedder.linear_1")||
     !linear(time.p,"proj_out",open,map,root,"time_embedder.linear_2"))return false;
  Owned refiner(PyObject_GetAttrString(model,"token_refiner"));
  Owned refiners(refiner.p?PyObject_GetAttrString(refiner.p,"blocks"):nullptr);
  if(!refiners.p)return false;
  for(int i=0;i<2;++i) {
    Owned key(PyLong_FromLong(i)),block(PyObject_GetItem(refiners.p,key.p));
    if(!block.p||!h3_attention_mlp(block.p,open,map,root,
        "token_refiner.refiner_blocks."+std::to_string(i)+"."))return false;
  }
  if(!linear(refiner.p,"final_norm",open,map,root,"token_refiner.final_norm",false))return false;
  Owned blocks(PyObject_GetAttrString(model,"blocks"));
  if(!blocks.p||PyObject_Length(blocks.p)!=50)return false;
  for(int i=0;i<50;++i) {
    Owned key(PyLong_FromLong(i)),block(PyObject_GetItem(blocks.p,key.p));
    const std::string prefix="transformer_blocks."+std::to_string(i)+".";
    if(!block.p||!h3_attention_mlp(block.p,open,map,root,prefix))return false;
    Owned adaln(PyObject_GetAttrString(block.p,"adaln_proj"));
    if(!adaln.p||!linear(adaln.p,"linear",open,map,root,prefix+"adaln_proj.linear"))return false;
    std::printf("original_trajectory_loaded_block=%d\n",i);std::fflush(stdout);
  }
  Owned head(PyObject_GetAttrString(model,"final_layer"));
  Owned adaln(head.p?PyObject_GetAttrString(head.p,"adaln_proj"):nullptr);
  return head.p&&adaln.p&&linear(head.p,"norm",open,map,root,"norm_out.norm",false)&&
    linear(adaln.p,"linear",open,map,root,"norm_out.linear")&&
    linear(head.p,"video_out",open,map,root,"proj_out")&&
    linear(head.p,"audio_out",open,map,root,"audio_proj_out");
}

static PyObject *observe(PyObject *self,PyObject *args) {
  int step;PyObject *video,*audio;
  if(!PyArg_ParseTuple(args,"iOO",&step,&video,&audio))return nullptr;
  auto *path=static_cast<const char*>(PyCapsule_GetPointer(self,"trajectory_output"));
  Owned torch(PyImport_ImportModule("torch")),u8(torch.p?PyObject_GetAttrString(torch.p,"uint8"):nullptr);
  if(!path||!u8.p||!save(video,u8.p,std::string(path)+"/video-step-"+std::to_string(step)+".f32",592LL*96*4)||
     !save(audio,u8.p,std::string(path)+"/audio-step-"+std::to_string(step)+".f32",414LL*32*4))return nullptr;
  std::printf("original_trajectory_update=%d video_rows=592 audio_rows=414\n",step+1);std::fflush(stdout);
  Py_RETURN_NONE;
}

static PyMethodDef observer={"observe",observe,METH_VARARGS,"Retain literal original step outputs."};

static bool packed_identity(PyObject *torch,PyObject *packed,PyObject *f32,PyObject *u8,
                            const char *input,const char *out) {
  const std::string base(input);
  // Compare original literal positions at the same F32 conversion boundary as
  // native capture. Keep original F64 positions for the actual original model.
  auto *positions=PyDict_GetItemString(packed,"img_position_ids");
  auto *boundaries=PyDict_GetItemString(packed,"cu_seqlens");
  auto *tags=PyDict_GetItemString(packed,"token_tags");
  Owned p(positions?PyObject_CallMethod(positions,"to","O",f32):nullptr);
  if(!p.p||!boundaries||!tags||!save(p.p,u8,std::string(out)+"/positions.f32",1024LL*3*4)||
     !save(boundaries,u8,std::string(out)+"/documents.i32",12))return false;
  for(const char *file:{"positions.f32","documents.i32"}) {
    std::ifstream a(base+"/"+file,std::ios::binary),b(std::string(out)+"/"+file,std::ios::binary);
    const std::vector<char> x((std::istreambuf_iterator<char>(a)),{}),y((std::istreambuf_iterator<char>(b)),{});
    if(!a||!b||x!=y){PyErr_Format(PyExc_ValueError,"literal original packing differs at %s",file);return false;}
  }
  Owned clamp_fn(PyObject_GetAttrString(tags,"clamp")),empty(PyTuple_New(0)),kw(Py_BuildValue("{s:i}","min",0));
  Owned clamped(clamp_fn.p?PyObject_Call(clamp_fn.p,empty.p,kw.p):nullptr);
  Owned values(clamped.p?PyObject_CallMethod(clamped.p,"tolist",nullptr):nullptr);
  std::ifstream file(base+"/combined.i32",std::ios::binary);
  if(!values.p||PyList_Size(values.p)!=1024||!file)return false;
  for(int row=0;row<1024;++row) {
    int32_t value;file.read(reinterpret_cast<char*>(&value),4);
    if(!file||value!=PyLong_AsLong(PyList_GetItem(values.p,row))) {
      PyErr_SetString(PyExc_ValueError,"literal original initial modality ownership differs");return false;
    }
  }
  (void)torch;
  return true;
}

static PyObject *schedule(PyObject *module,const char *input,int modality,int original_points) {
  Owned fn(PyObject_GetAttrString(module,"minimax_h3_time_shift_sigmas")),args(PyTuple_New(0));
  Owned kw(Py_BuildValue("{s:i,s:d}","num_steps",original_points?original_points:21,"shift_scale",modality?3.0:12.0));
  Owned original(fn.p?PyObject_Call(fn.p,args.p,kw.p):nullptr);
  if(!original.p)return nullptr;
  if(original_points) {
    std::printf("original_schedule modality=%s points=%d literal_native_schedule=false quality_schedule_control=true\n",modality?"audio":"video",original_points);
    return Py_NewRef(original.p);
  }
  // Use literal native sigma points for matched integration. Independently
  // report the installed original schedule's differences, without substituting
  // a newly generated seed or claiming bitwise trajectory equivalence.
  std::ifstream file(std::string(input)+"/flow-table.f32",std::ios::binary);
  std::vector<float> flow(160);file.read(reinterpret_cast<char*>(flow.data()),640);
  if(!file||file.peek()!=std::ifstream::traits_type::eof())return nullptr;
  Owned sigmas(PyList_New(21));
  for(int step=0;step<20;++step)PyList_SET_ITEM(sigmas.p,step,PyFloat_FromDouble(flow[step*8+modality*4]));
  PyList_SET_ITEM(sigmas.p,20,PyFloat_FromDouble(0.0));
  if(PyObject_Length(original.p)!=21)return nullptr;
  double max_difference=0;int exact=0;
  for(int step=0;step<21;++step) {
    const double a=PyFloat_AsDouble(PyList_GetItem(sigmas.p,step));
    const double b=PyFloat_AsDouble(PyList_GetItem(original.p,step));
    exact+=a==b;max_difference=std::max(max_difference,std::abs(a-b));
  }
  std::printf("original_schedule modality=%s points=21 matched_points=%d maximum_difference=%.17g literal_native_schedule=true\n",
    modality?"audio":"video",exact,max_difference);
  return Py_NewRef(sigmas.p);
}

static int execute(const char *root,const char *input,const char *out,int original_points) {
  Owned torch(PyImport_ImportModule("torch")),safe(PyImport_ImportModule("safetensors"));
  Owned dist(PyImport_ImportModule("vllm.distributed")),cfgmod(PyImport_ImportModule("vllm.config"));
  Owned cfgcls(cfgmod.p?PyObject_GetAttrString(cfgmod.p,"VllmConfig"):nullptr),cfg(cfgcls.p?PyObject_CallNoArgs(cfgcls.p):nullptr);
  Owned scope(cfg.p?PyObject_CallMethod(cfgmod.p,"set_current_vllm_config","O",cfg.p):nullptr);
  Owned entered(scope.p?PyObject_CallMethod(scope.p,"__enter__",nullptr):nullptr);
  if(!torch.p||!safe.p||!dist.p||!entered.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),f32(PyObject_GetAttrString(torch.p,"float32")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned init(PyObject_GetAttrString(dist.p,"init_distributed_environment")),empty(PyTuple_New(0));
  Owned opts(Py_BuildValue("{s:i,s:i,s:i,s:s,s:s}","world_size",1,"rank",0,"local_rank",0,
    "distributed_init_method","file:///work/rendezvous","backend","gloo"));
  Owned world(init.p?PyObject_Call(init.p,empty.p,opts.p):nullptr);
  Owned parallel(PyObject_CallMethod(dist.p,"initialize_model_parallel","ii",1,1));
  if(!grad.p||!threads.p||!dtype.p||!world.p||!parallel.p)return -1;
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer"));
  Owned metadata(json_file(std::string(root)+"/config.json"));
  Owned types(PyImport_ImportModule("types")),ns(types.p?PyObject_GetAttrString(types.p,"SimpleNamespace"):nullptr);
  Owned pcargs(PyTuple_New(0)),pckw(Py_BuildValue("{s:i}","ulysses_degree",1));
  Owned pc(ns.p?PyObject_Call(ns.p,pcargs.p,pckw.p):nullptr);
  Owned odkw(metadata.p&&pc.p?Py_BuildValue("{s:O,s:O}","tf_model_config",metadata.p,"parallel_config",pc.p):nullptr);
  Owned od(odkw.p?PyObject_Call(ns.p,pcargs.p,odkw.p):nullptr);
  Owned cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTModel"):nullptr);
  // Allocate once on GPU; constructing ~66 GB on CPU and then copying the
  // whole model would exceed this unified-memory machine's capacity.
  Owned device(PyObject_CallMethod(torch.p,"device","s","cuda"));
  Owned device_entered(device.p?PyObject_CallMethod(device.p,"__enter__",nullptr):nullptr);
  Owned ctor(od.p?PyTuple_Pack(2,od.p,Py_None):nullptr);
  Owned model(device_entered.p&&cls.p&&ctor.p?PyObject_CallObject(cls.p,ctor.p):nullptr);
  Owned device_closed(device.p?PyObject_CallMethod(device.p,"__exit__","OOO",Py_None,Py_None,Py_None):nullptr);
  Owned index(json_file(std::string(root)+"/diffusion_pytorch_model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  if(!model.p||!device_closed.p||!map||!open.p||!load_model(model.p,open.p,map,root))return -1;
  Owned rope(PyObject_GetAttrString(model.p,"rope")),freq(rope.p?PyObject_GetAttrString(rope.p,"inv_freq"):nullptr);
  Owned f0(literal_tensor(torch.p,f32.p,(std::string(root)+"/rope.inv_freq.f32").c_str(),1,16,4));
  Owned f(f0.p?PyObject_CallMethod(f0.p,"view","i",16):nullptr),copied(freq.p&&f.p?PyObject_CallMethod(freq.p,"copy_","O",f.p):nullptr);
  Owned checked(copied.p?PyObject_CallMethod(model.p,"post_load_weights",nullptr):nullptr);
  Owned evaluated(checked.p?PyObject_CallMethod(model.p,"eval",nullptr):nullptr);
  if(!evaluated.p)return -1;
  Owned packing(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.packed_sequence"));
  Owned packfn(packing.p?PyObject_GetAttrString(packing.p,"minimax_h3_packed_sequence"):nullptr);
  Owned packkw(Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:O}","text_len",3,"latent_t",37,
    "latent_h",8,"latent_w",8,"audio_t",207,"include_keyframe_cond",Py_False));
  Owned packed(packfn.p?PyObject_Call(packfn.p,empty.p,packkw.p):nullptr);
  if(!packed.p||!packed_identity(torch.p,packed.p,f32.p,u8.p,input,out))return -1;
  const std::string base(input);
  Owned caption0(literal_bf16(torch.p,bf16.p,(base+"/encoded.bf16").c_str(),3,5120));
  Owned caption(caption0.p?PyObject_CallMethod(caption0.p,"view","ii",3,5120):nullptr);
  Owned video0(literal_tensor(torch.p,f32.p,(base+"/video-state.f32").c_str(),592,96,4));
  Owned audio0(literal_tensor(torch.p,f32.p,(base+"/audio-state.f32").c_str(),414,32,4));
  Owned video(video0.p?PyObject_CallMethod(video0.p,"view","ii",592,96):nullptr);
  Owned audio(audio0.p?PyObject_CallMethod(audio0.p,"view","ii",414,32):nullptr);
  Owned denoise(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.denoise_loop"));
  Owned branchcls(denoise.p?PyObject_GetAttrString(denoise.p,"MiniMaxH3DenoiseBranch"):nullptr);
  auto *tags=PyDict_GetItemString(packed.p,"token_tags");
  Owned branchkw(caption.p&&tags?Py_BuildValue("{s:O,s:O,s:O,s:O}","packed",packed.p,
    "text_embeddings",caption.p,"token_tags",tags,"device",device.p):nullptr);
  Owned branch(branchcls.p&&branchkw.p?PyObject_Call(branchcls.p,empty.p,branchkw.p):nullptr);
  Owned timemod(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.time_request"));
  Owned sv(timemod.p?schedule(timemod.p,input,0,original_points):nullptr),sa(timemod.p?schedule(timemod.p,input,1,original_points):nullptr);
  Owned capsule(PyCapsule_New(const_cast<char*>(out),"trajectory_output",nullptr));
  Owned callback(capsule.p?PyCFunction_New(&observer,capsule.p):nullptr);
  Owned fn(denoise.p?PyObject_GetAttrString(denoise.p,"minimax_h3_denoise_loop"):nullptr);
  Owned kw(branch.p&&sv.p&&sa.p&&video.p&&audio.p&&callback.p?Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
    "model",model.p,"positive",branch.p,"initial_video_rows",video.p,"initial_audio_rows",audio.p,
    "keyframe_cond_rows",Py_None,"sigmas_video",sv.p,"sigmas_audio",sa.p,"device",device.p,"on_step",callback.p):nullptr);
  if(!fn.p||!kw.p)return -1;
  const int updates=original_points?original_points-1:20;
  std::printf("original_trajectory_start updates=%d literal_noise=true literal_caption=true encoder_omitted=true\n",updates);std::fflush(stdout);
  Owned terminal(PyObject_Call(fn.p,empty.p,kw.p));
  Owned tokens(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.packed_tokens"));
  Owned vfn(tokens.p?PyObject_GetAttrString(tokens.p,"minimax_h3_unpatchify_video_tokens"):nullptr);
  Owned afn(tokens.p?PyObject_GetAttrString(tokens.p,"minimax_h3_unpack_audio_tokens"):nullptr);
  if(!terminal.p||!vfn.p||!afn.p)return -1;
  Owned vargs(PyTuple_Pack(1,PyTuple_GetItem(terminal.p,0))),aargs(PyTuple_Pack(1,PyTuple_GetItem(terminal.p,1)));
  Owned vkw(Py_BuildValue("{s:[iiii],s:[iii]}","latent_shape",37,4,4,24,"patch_size",1,2,2));
  Owned akw(Py_BuildValue("{s:i,s:i}","audio_t",414,"audio_channel",2));
  Owned vl(PyObject_Call(vfn.p,vargs.p,vkw.p)),al(PyObject_Call(afn.p,aargs.p,akw.p));
  if(!vl.p||!al.p||!save(vl.p,u8.p,std::string(out)+"/video-latent.f32",24LL*37*8*8*4)||
     !save(al.p,u8.p,std::string(out)+"/audio-latent.f32",2LL*32*207*4))return -1;
  Owned closed(PyObject_CallMethod(scope.p,"__exit__","OOO",Py_None,Py_None,Py_None));
  std::printf("original_trajectory_terminal updates=%d video_latent=1,24,37,8,8 audio_latent=2,32,207 media_quality=unassessed\n",updates);
  return closed.p?0:-1;
}

int main(int argc,char **argv) {
  if(argc!=4&&argc!=5)return 2;
  int points=0;
  if(argc==5) {
    char *end=nullptr;const long parsed=std::strtol(argv[4],&end,10);
    if(end==argv[4]||*end||parsed<2||parsed>50)return 2;
    points=static_cast<int>(parsed);
  }
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],points);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
