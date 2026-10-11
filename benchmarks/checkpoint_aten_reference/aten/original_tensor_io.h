#pragma once
// Test-only bounded original-checkpoint tensor I/O. No model arithmetic.
#include <Python.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *v=nullptr):p(v) {}
  ~Owned(){Py_XDECREF(p);}
  Owned(const Owned &)=delete;
};

inline PyObject *json_file(const std::string &path) {
  std::ifstream file(path);
  if(!file){PyErr_SetString(PyExc_OSError,"original metadata unavailable");return nullptr;}
  const std::string bytes((std::istreambuf_iterator<char>(file)),{});
  Owned json(PyImport_ImportModule("json"));
  return json.p?PyObject_CallMethod(json.p,"loads","s",bytes.c_str()):nullptr;
}

inline PyObject *weight(PyObject *open,PyObject *map,const char *root,const char *name,
                        PyObject *ids=nullptr) {
  auto *entry=PyDict_GetItemString(map,name);
  const char *shard=entry?PyUnicode_AsUTF8(entry):nullptr;
  if(!shard){PyErr_Format(PyExc_KeyError,"missing original tensor %s",name);return nullptr;}
  Owned path(PyUnicode_FromString((std::string(root)+"/"+shard).c_str()));
  Owned args(PyTuple_Pack(1,path.p));
  Owned kw(Py_BuildValue("{s:s,s:s}","framework","pt","device","cpu"));
  Owned reader(PyObject_Call(open,args.p,kw.p));
  Owned tensor(reader.p?PyObject_CallMethod(reader.p,"get_tensor","s",name):nullptr);
  // Select embedding rows before the copy; never instantiate a model/state dict.
  Owned selected(tensor.p?(ids?PyObject_CallMethod(tensor.p,"index_select","iO",0,ids):
    PyObject_CallMethod(tensor.p,"clone",nullptr)):nullptr);
  return selected.p?PyObject_CallMethod(selected.p,"to","s","cuda"):nullptr;
}

inline bool save(PyObject *tensor,PyObject *u8,const std::string &path,int64_t extent) {
  Owned cpu(PyObject_CallMethod(tensor,"cpu",nullptr));
  Owned contiguous(cpu.p?PyObject_CallMethod(cpu.p,"contiguous",nullptr):nullptr);
  Owned raw(contiguous.p?PyObject_CallMethod(contiguous.p,"view","O",u8):nullptr);
  Owned array(raw.p?PyObject_CallMethod(raw.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p||PyBytes_Size(bytes.p)!=extent){PyErr_SetString(PyExc_ValueError,"component extent differs");return false;}
  auto *file=std::fopen(path.c_str(),"wbx");
  if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,path.c_str());return false;}
  const bool wrote=std::fwrite(PyBytes_AsString(bytes.p),1,extent,file)==static_cast<size_t>(extent);
  const bool closed=std::fclose(file)==0;
  if(!wrote||!closed){PyErr_SetString(PyExc_OSError,"component write failed");return false;}
  return true;
}

inline PyObject *literal_bf16(PyObject *torch,PyObject *dtype,const char *path,int rows,int width) {
  std::ifstream file(path,std::ios::binary);
  const std::vector<char> data((std::istreambuf_iterator<char>(file)),{});
  if(!file||static_cast<int64_t>(data.size())!=int64_t(rows)*width*2){
    PyErr_SetString(PyExc_ValueError,"literal component input extent differs");return nullptr;
  }
  Owned bytes(PyByteArray_FromStringAndSize(data.data(),data.size()));
  Owned factory(PyObject_GetAttrString(torch,"frombuffer"));
  Owned args(bytes.p?PyTuple_Pack(1,bytes.p):nullptr),kw(Py_BuildValue("{s:O}","dtype",dtype));
  Owned flat(factory.p&&args.p?PyObject_Call(factory.p,args.p,kw.p):nullptr);
  Owned shape(flat.p?PyObject_CallMethod(flat.p,"view","iii",1,rows,width):nullptr);
  return shape.p?PyObject_CallMethod(shape.p,"to","s","cuda"):nullptr;
}
