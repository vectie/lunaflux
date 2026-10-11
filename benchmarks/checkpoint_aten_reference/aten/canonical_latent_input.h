// Offline CPython tensor input shared by original VAE replay bridges.
// No production package includes this header or links Torch.
#pragma once
#include <Python.h>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
struct Owned {
  PyObject *p;
  explicit Owned(PyObject *value):p(value){}
  ~Owned(){Py_XDECREF(p);}
  Owned(const Owned&)=delete;
};
static PyObject *read_canonical_latent(
    PyObject *torch,const char *path,std::initializer_list<int> dimensions) {
  size_t elements=1;
  for(auto dimension:dimensions)elements*=static_cast<size_t>(dimension);
  std::ifstream stream(path,std::ios::binary);
  if(!stream){PyErr_SetString(PyExc_OSError,"latent input cannot be opened");return nullptr;}
  std::string payload((std::istreambuf_iterator<char>(stream)),{});
  if(payload.size()!=elements*sizeof(float)) {
    PyErr_SetString(PyExc_ValueError,"canonical F32 latent extent mismatch");return nullptr;
  }
  Owned buffer(PyByteArray_FromStringAndSize(payload.data(),payload.size()));
  Owned fn(PyObject_GetAttrString(torch,"frombuffer")),f32(PyObject_GetAttrString(torch,"float32"));
  Owned args(buffer.p?PyTuple_Pack(1,buffer.p):nullptr),kwargs(f32.p?Py_BuildValue("{s:O}","dtype",f32.p):nullptr);
  Owned borrowed(fn.p&&args.p&&kwargs.p?PyObject_Call(fn.p,args.p,kwargs.p):nullptr);
  // frombuffer borrows the bytearray. Clone before its owner leaves scope.
  Owned input(borrowed.p?PyObject_CallMethod(borrowed.p,"clone",nullptr):nullptr);
  Owned shape(PyTuple_New(dimensions.size()));
  if(!input.p||!shape.p)return nullptr;
  size_t index=0;
  for(auto dimension:dimensions) {
    auto *item=PyLong_FromLong(dimension);
    if(!item)return nullptr;
    PyTuple_SET_ITEM(shape.p,index++,item);
  }
  return PyObject_CallMethod(input.p,"view","O",shape.p);
}
static int canonical_input_probe(
    const char *path,std::initializer_list<int> dimensions) {
  Owned torch(PyImport_ImportModule("torch"));
  Owned input(torch.p?read_canonical_latent(torch.p,path,dimensions):nullptr);
  Owned array(input.p?PyObject_CallMethod(input.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p)return -1;
  std::ifstream stream(path,std::ios::binary);
  std::string payload((std::istreambuf_iterator<char>(stream)),{});
  if(payload.size()!=static_cast<size_t>(PyBytes_Size(bytes.p))||
     payload!=std::string(PyBytes_AsString(bytes.p),PyBytes_Size(bytes.p))) {
    PyErr_SetString(PyExc_AssertionError,"canonical latent loader changed F32 bytes");return -1;
  }
  return 0;
}
