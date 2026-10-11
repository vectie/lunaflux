// Original sparse attention on the literal production rotary query and KV.
// No repeated rotary/quantization, replacement math, or changed original kernel.
#include "original_tensor_io.h"
#include <cmath>

static int execute(const char *inference,const char *input,const char *out) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model")),kernel(PyImport_ImportModule("kernel"));
  if(!torch.p||!model.p||!kernel.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),f32(PyObject_GetAttrString(torch.p,"float32")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned qflat(literal_bf16(torch.p,bf16.p,(std::string(input)+"/q-rope.bf16").c_str(),144,32768));
  Owned q(qflat.p?PyObject_CallMethod(qflat.p,"view","iiii",1,144,64,512):nullptr);
  Owned kv(literal_bf16(torch.p,bf16.p,(std::string(input)+"/kv-sim.bf16").c_str(),144,512));
  Owned sinkflat(literal_tensor(torch.p,f32.p,(std::string(input)+"/sink.f32").c_str(),1,64,4));
  Owned sink(sinkflat.p?PyObject_CallMethod(sinkflat.p,"view","i",64):nullptr);
  Owned window(PyObject_GetAttrString(model.p,"get_window_topk_idxs")),attention(PyObject_GetAttrString(kernel.p,"sparse_attn"));
  Owned args(Py_BuildValue("(iiii)",128,1,144,0));
  Owned indices(window.p?PyObject_CallObject(window.p,args.p):nullptr);
  Owned gpu(indices.p?PyObject_CallMethod(indices.p,"to","s","cuda"):nullptr);
  if(!grad.p||!threads.p||!q.p||!kv.p||!sink.p||!gpu.p||!attention.p)return -1;
  Owned operands(Py_BuildValue("(OOOOd)",q.p,kv.p,sink.p,gpu.p,1.0/std::sqrt(512.0)));
  Owned result(PyObject_CallObject(attention.p,operands.p));
  if(!result.p||!save(result.p,u8.p,std::string(out)+"/attention.bf16",144LL*32768*2))return -1;
  std::puts("original_window_literal_operands rows=144 heads=64 original_sparse_attention=true");
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=4)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
