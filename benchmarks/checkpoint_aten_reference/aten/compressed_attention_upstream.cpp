// Offline original sparse attention on literal rotary/simulated cache operands.
// No replacement arithmetic, weight casts, or whole-model parity assertion.
#include "original_tensor_io.h"
#include <cmath>

static int execute(const char *inference,const char *input,const char *output) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned kernel(PyImport_ImportModule("kernel"));
  if(!torch.p||!model.p||!kernel.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),f32(PyObject_GetAttrString(torch.p,"float32"));
  Owned u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  if(!grad.p||!threads.p||!dtype.p||!u8.p||!f32.p)return -1;
  Owned qflat(literal_bf16(torch.p,bf16.p,(std::string(input)+"/q-rope.bf16").c_str(),144,32768));
  Owned q(qflat.p?PyObject_CallMethod(qflat.p,"view","iiii",1,144,64,512):nullptr);
  Owned kv(literal_bf16(torch.p,bf16.p,(std::string(input)+"/kv-sim.bf16").c_str(),144,512));
  Owned compressed(literal_bf16(torch.p,bf16.p,(std::string(input)+"/compressed.bf16").c_str(),1,512));
  Owned sink_flat(literal_tensor(torch.p,f32.p,(std::string(input)+"/sink.f32").c_str(),1,64,4));
  Owned sink(sink_flat.p?PyObject_CallMethod(sink_flat.p,"view","i",64):nullptr);
  Owned win_fn(PyObject_GetAttrString(model.p,"get_window_topk_idxs"));
  Owned comp_fn(PyObject_GetAttrString(model.p,"get_compress_topk_idxs"));
  Owned attention(PyObject_GetAttrString(kernel.p,"sparse_attn"));
  Owned cat(PyObject_GetAttrString(torch.p,"cat"));
  Owned wa(Py_BuildValue("(iiii)",128,1,144,0)),ca(Py_BuildValue("(iiiii)",128,1,144,0,144));
  Owned wi(win_fn.p?PyObject_CallObject(win_fn.p,wa.p):nullptr);
  Owned ci(comp_fn.p?PyObject_CallObject(comp_fn.p,ca.p):nullptr);
  Owned il(wi.p&&ci.p?PyTuple_Pack(2,wi.p,ci.p):nullptr);
  Owned ia(il.p?Py_BuildValue("(Oi)",il.p,-1):nullptr);
  Owned indices(cat.p&&ia.p?PyObject_CallObject(cat.p,ia.p):nullptr);
  Owned gpu_indices(indices.p?PyObject_CallMethod(indices.p,"to","s","cuda"):nullptr);
  Owned kl(kv.p&&compressed.p?PyTuple_Pack(2,kv.p,compressed.p):nullptr);
  Owned ka(kl.p?Py_BuildValue("(Oi)",kl.p,1):nullptr);
  Owned keys(cat.p&&ka.p?PyObject_CallObject(cat.p,ka.p):nullptr);
  if(!q.p||!sink.p||!gpu_indices.p||!keys.p||!attention.p)return -1;
  Owned args(Py_BuildValue("(OOOOd)",q.p,keys.p,sink.p,gpu_indices.p,1.0/std::sqrt(512.0)));
  Owned result(PyObject_CallObject(attention.p,args.p));
  if(!result.p||!save(result.p,u8.p,std::string(output)+"/attention.bf16",int64_t(144)*32768*2)||
     !save(gpu_indices.p,u8.p,std::string(output)+"/indices.i32",int64_t(144)*129*4))return -1;
  std::printf("original_compressed_attention_complete rows=144 ratio=128 keys=129 arithmetic=original_functions literal_inputs=true full_model=false\n");
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=4)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
