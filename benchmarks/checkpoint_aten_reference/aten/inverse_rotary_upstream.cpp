// Offline original inverse RoPE on literal production sparse-attention rows.
// No attention, projection, normalization, quantization, or model execution.
#include "original_tensor_io.h"

static int execute(const char *inference,const char *input,const char *frequencies,const char *out) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  if(!torch.p||!model.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16"));
  Owned complex(PyObject_GetAttrString(torch.p,"complex64")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  if(!grad.p||!threads.p||!bf16.p||!complex.p||!u8.p)return -1;
  Owned x0(literal_bf16(torch.p,bf16.p,input,144,32768));
  Owned f0(literal_tensor(torch.p,complex.p,frequencies,144,32,8));
  Owned x(x0.p?PyObject_CallMethod(x0.p,"view","iiii",1,144,64,512):nullptr);
  Owned f(f0.p?PyObject_CallMethod(f0.p,"view","ii",144,32):nullptr);
  Owned rotary(PyObject_GetAttrString(model.p,"apply_rotary_emb"));
  if(!x.p||!f.p||!rotary.p)return -1;
  for(int row=0;row<144;++row) {
    Owned one(PyObject_CallMethod(x.p,"narrow","iii",1,row,1));
    Owned suffix(one.p?PyObject_CallMethod(one.p,"narrow","iii",-1,448,64):nullptr);
    Owned frequency(PyObject_CallMethod(f.p,"narrow","iii",0,row,1));
    Owned args(suffix.p&&frequency.p?PyTuple_Pack(3,suffix.p,frequency.p,Py_True):nullptr);
    Owned result(args.p?PyObject_CallObject(rotary.p,args.p):nullptr);
    if(!result.p)return -1;
  }
  if(!save(x.p,u8.p,out,144LL*32768*2))return -1;
  std::puts("original_inverse_rotary_complete rows=144 heads=64 head_dim=512 rotary_suffix=64 one_row_geometry=true literal_native_attention=true literal_original_frequencies=true full_model=false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=5)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
