// Offline bridge to the installed ORIGINAL H3 AudioVAE adapter. Model loading,
// destandardization and decoding are original code, not a new arithmetic port.
#include "canonical_latent_input.h"
#include <cstdio>
static int execute(const char *root,const char *source,const char *destination) {
  Owned torch(PyImport_ImportModule("torch"));
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.vae"));
  if(!torch.p||!module.p)return -1;
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned input(read_canonical_latent(torch.p,source,{2,32,207}));
  if(!threads.p||!input.p)return -1;
  Owned cls(PyObject_GetAttrString(module.p,"MiniMaxH3AudioVAE"));
  Owned path(PyUnicode_FromString(root));
  Owned args(path.p?PyTuple_Pack(1,path.p):nullptr);
  Owned device(PyObject_CallMethod(torch.p,"device","s","cuda"));
  Owned kwargs(device.p?Py_BuildValue("{s:O}","device",device.p):nullptr);
  Owned model(cls.p&&args.p&&kwargs.p?PyObject_Call(cls.p,args.p,kwargs.p):nullptr);
  Owned moved(PyObject_CallMethod(input.p,"to","s","cuda"));
  Owned waveform(model.p&&moved.p?PyObject_CallMethod(model.p,"decode_latent","O",moved.p):nullptr);
  if(!waveform.p)return -1;
  Owned cpu(PyObject_CallMethod(waveform.p,"cpu",nullptr));
  Owned array(cpu.p?PyObject_CallMethod(cpu.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  Owned dimensions(cpu.p?PyObject_GetAttrString(cpu.p,"shape"):nullptr);
  Owned repr(dimensions.p?PyObject_Repr(dimensions.p):nullptr);
  if(!bytes.p||!repr.p)return -1;
  // The original adapter canonicalizes stereo to [1,2,time].
  if(PyBytes_Size(bytes.p)!=2*165600*4) {
    PyErr_SetString(PyExc_ValueError,"original audio output extent mismatch");return -1;
  }
  auto *file=std::fopen(destination,"wbx");
  if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,destination);return -1;}
  const auto count=PyBytes_Size(bytes.p);
  const bool written=std::fwrite(PyBytes_AsString(bytes.p),1,count,file)==static_cast<size_t>(count);
  const bool closed=std::fclose(file)==0;
  if(!written||!closed){PyErr_SetString(PyExc_OSError,"original waveform write failed");return -1;}
  std::printf("original_audio_vae=true standardized_latent=[2,32,207] dtype=float32 output_shape=%s output_bytes=%zd\n",PyUnicode_AsUTF8(repr.p),count);
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=3&&argc!=4){std::fputs("ORIGINAL_COMPONENT CANONICAL_LATENT_F32 NEW_WAVEFORM_F32 | --input-probe LATENT_F32\n",stderr);return 2;}
  const bool probe=std::string(argv[1])=="--input-probe";
  if(!probe&&argc!=4)return 2;
  Py_Initialize();
  const int result=probe?canonical_input_probe(argv[2],{2,32,207}):execute(argv[1],argv[2],argv[3]);
  if(PyErr_Occurred())PyErr_Print();
  if(!result&&probe)std::puts("original_audio_latent_input_probe_passed cpu_only=true bytes_unchanged=true");
  const int finalized=Py_FinalizeEx();
  return result||finalized<0?1:0;
}
