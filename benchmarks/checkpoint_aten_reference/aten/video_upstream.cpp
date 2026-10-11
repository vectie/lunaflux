// Test-only CPython bridge to the ORIGINAL VideoVAE modules. No alternative
// decoder, tensor math implementation, or Python automation is supplied here.
#include "canonical_latent_input.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
struct VideoShape { int time=37, height=2, width=2; };
static PyObject* tensor(PyObject *torch,PyObject *values) {
  Owned fn(PyObject_GetAttrString(torch,"tensor")),args(PyTuple_Pack(1,values));
  Owned kwargs(Py_BuildValue("{s:s}","device","cuda"));
  return fn.p&&args.p&&kwargs.p?PyObject_Call(fn.p,args.p,kwargs.p):nullptr;
}
// The joint diagnostic exports standardized canonical NCTHW, not decoder
// activations or destandardized latents. Loading stays CPU-only until replay.
static PyObject *read_latent(PyObject *torch,const char *path,VideoShape shape) {
  return read_canonical_latent(torch,path,{1,24,shape.time,shape.height,shape.width});
}
static int input_probe(const char *path,VideoShape shape) {
  if(canonical_input_probe(path,{1,24,shape.time,shape.height,shape.width}))return -1;
  std::puts("original_video_latent_input_probe_passed cpu_only=true shape=NCTHW bytes_unchanged=true");
  return 0;
}
static int execute(const char *root,const char *destination,const char *source,VideoShape geometry) {
  auto *paths=PySys_GetObject("path");
  Owned directory(PyUnicode_FromString(root));
  if(!paths||!directory.p||PyList_Insert(paths,0,directory.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),json(PyImport_ImportModule("json"));
  Owned module(PyImport_ImportModule("video_vae.minimax_h3_video_vae"));
  if(!torch.p||!json.p||!module.p)return -1;
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned cls(PyObject_GetAttrString(module.p,"MiniMaxH3VideoVAE"));
  const std::string component=std::string(root)+"/video_vae";
  Owned model(cls.p?PyObject_CallMethod(cls.p,"from_pretrained","s",component.c_str()):nullptr);
  if(!threads.p||!model.p)return -1;
  Owned moved(PyObject_CallMethod(model.p,"to","s","cuda"));
  if(!moved.p)return -1;
  std::ifstream stream(component+"/config.json");
  std::string text((std::istreambuf_iterator<char>(stream)),{});
  Owned config(PyObject_CallMethod(json.p,"loads","s",text.c_str()));
  if(!config.p||!PyDict_Check(config.p))return -1;
  Owned mean(tensor(torch.p,PyDict_GetItemString(config.p,"latents_mean")));
  Owned stddev(tensor(torch.p,PyDict_GetItemString(config.p,"latents_std")));
  Owned mean_view(mean.p?PyObject_CallMethod(mean.p,"view","iiiii",1,24,1,1,1):nullptr);
  Owned std_view(stddev.p?PyObject_CallMethod(stddev.p,"view","iiiii",1,24,1,1,1):nullptr);
  Owned shape(Py_BuildValue("(iiiii)",1,24,geometry.time,geometry.height,geometry.width)),args(PyTuple_Pack(1,shape.p));
  Owned zeros(PyObject_GetAttrString(torch.p,"zeros"));
  Owned f32(PyObject_GetAttrString(torch.p,"float32"));
  Owned kwargs(Py_BuildValue("{s:s,s:O}","device","cuda","dtype",f32.p));
  Owned cpu_input(source?read_latent(torch.p,source,geometry):nullptr);
  Owned input(source?(cpu_input.p?PyObject_CallMethod(cpu_input.p,"to","s","cuda"):nullptr):
    (zeros.p?PyObject_Call(zeros.p,args.p,kwargs.p):nullptr));
  Owned scaled(input.p&&std_view.p?PyNumber_Multiply(input.p,std_view.p):nullptr);
  Owned latent(scaled.p&&mean_view.p?PyNumber_Add(scaled.p,mean_view.p):nullptr);
  if(!latent.p)return -1;
  Owned inference(PyObject_CallMethod(torch.p,"inference_mode",nullptr));
  Owned entered(inference.p?PyObject_CallMethod(inference.p,"__enter__",nullptr):nullptr);
  Owned half(PyObject_GetAttrString(torch.p,"float16"));
  Owned autocast_fn(PyObject_GetAttrString(torch.p,"autocast"));
  Owned autocast_args(Py_BuildValue("(s)","cuda"));
  Owned autocast_kwargs(Py_BuildValue("{s:O}","dtype",half.p));
  Owned autocast(autocast_fn.p?PyObject_Call(autocast_fn.p,autocast_args.p,autocast_kwargs.p):nullptr);
  Owned autocast_enter(autocast.p?PyObject_CallMethod(autocast.p,"__enter__",nullptr):nullptr);
  if(!entered.p||!autocast_enter.p)return -1;
  Owned decoded(PyObject_CallMethod(model.p,"decode_base","O",latent.p));
  Owned autocast_exit(PyObject_CallMethod(autocast.p,"__exit__","OOO",Py_None,Py_None,Py_None));
  Owned processor(PyObject_GetAttrString(model.p,"processor"));
  Owned frames(decoded.p&&processor.p?PyObject_CallMethod(processor.p,"revert_tensor","O",decoded.p):nullptr);
  if(!frames.p||!autocast_exit.p)return -1;
  // The original processor already returns contiguous [B,C,T,H,W]. Do not
  // invent another batch dimension or transpose its channel/time axes.
  Owned float_frames(PyObject_CallMethod(frames.p,"float",nullptr));
  Owned contiguous(float_frames.p?PyObject_CallMethod(float_frames.p,"contiguous",nullptr):nullptr);
  Owned cpu(contiguous.p?PyObject_CallMethod(contiguous.p,"cpu",nullptr):nullptr);
  Owned array(cpu.p?PyObject_CallMethod(cpu.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p||!PyBytes_Check(bytes.p))return -1;
  Owned dimensions(PyObject_GetAttrString(cpu.p,"shape")),repr(dimensions.p?PyObject_Repr(dimensions.p):nullptr);
  if(!repr.p)return -1;
  const auto count=PyBytes_Size(bytes.p);
  auto *file=std::fopen(destination,"wbx");
  if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,destination);return -1;}
  const bool written=std::fwrite(PyBytes_AsString(bytes.p),1,count,file)==static_cast<size_t>(count);
  const bool closed=std::fclose(file)==0;
  if(!written||!closed)return -1;
  std::printf("original_video_vae=true standardized_latent=[1,24,%d,%d,%d] latent_input=%s autocast=float16 output_shape=%s output_bytes=%zd\n",geometry.time,geometry.height,geometry.width,source?source:"zero",PyUnicode_AsUTF8(repr.p),count);
  Owned exited(PyObject_CallMethod(inference.p,"__exit__","OOO",Py_None,Py_None,Py_None));
  return exited.p?0:-1;
}
int main(int argc,char **argv) {
  if(argc!=3&&argc!=4&&argc!=6&&argc!=7){std::fputs("ORIGINAL_SOURCE_ROOT NEW_PIXELS_F32 [CANONICAL_STANDARDIZED_LATENT_F32] [LATENT_TIME LATENT_HEIGHT LATENT_WIDTH] | --input-probe LATENT_F32 [LATENT_TIME LATENT_HEIGHT LATENT_WIDTH]\n",stderr);return 2;}
  const bool probe=std::string(argv[1])=="--input-probe";
  if(probe&&(argc==4||argc==7))return 2;
  VideoShape shape;
  if(argc>=6) {
    int *fields[]={&shape.time,&shape.height,&shape.width};
    for(int i=0;i<3;++i) {
      char *end=nullptr;
      const long value=std::strtol(argv[argc-3+i],&end,10);
      // This diagnostic's 32-GiB budget does not admit arbitrary-resolution
      // decoder requests. Both real smoke/quality shapes fit these bounds.
      if(!end||*end||value<=0||value>(i==0?128:16))return 2;
      *fields[i]=static_cast<int>(value);
    }
  }
  const char *source=!probe&&(argc==4||argc==7)?argv[3]:nullptr;
  Py_Initialize();
  const int result=probe?input_probe(argv[2],shape):execute(argv[1],argv[2],source,shape);
  if(PyErr_Occurred())PyErr_Print();
  const int finalized=Py_FinalizeEx();
  return result||finalized<0?1:0;
}
