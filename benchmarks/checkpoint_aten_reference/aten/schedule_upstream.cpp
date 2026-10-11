// CPU-only diagnostic: call the installed original H3 sigma routine.
#include <Python.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
struct Owned {
  PyObject *p;
  explicit Owned(PyObject *value):p(value){}
  ~Owned(){Py_XDECREF(p);}
};
static int probe() {
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.time_request"));
  Owned fn(module.p?PyObject_GetAttrString(module.p,"minimax_h3_time_shift_sigmas"):nullptr);
  if(!fn.p)return -1;
  const int counts[]={2,5,10,20,50,1000};
  const int shifts[]={3,12};
  for(int count:counts)for(int shift:shifts) {
    Owned args(PyTuple_New(0));
    Owned kwargs(Py_BuildValue("{s:i,s:d}","num_steps",count,"shift_scale",double(shift)));
    Owned result(args.p&&kwargs.p?PyObject_Call(fn.p,args.p,kwargs.p):nullptr);
    Owned list(result.p?PySequence_List(result.p):nullptr);
    if(!list.p||PyList_Size(list.p)!=count)return -1;
    for(int index=0;index<count;++index) {
      float value=float(PyFloat_AsDouble(PyList_GetItem(list.p,index)));
      if(PyErr_Occurred())return -1;
      uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
      std::printf("%d\t%d\t%d\t%u\n",count,shift,index,bits);
    }
  }
  return 0;
}
int main(){Py_Initialize();int rc=probe();if(rc)PyErr_Print();Py_Finalize();return rc?1:0;}
