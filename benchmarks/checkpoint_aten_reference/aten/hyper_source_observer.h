#pragma once
// Offline inspection of the installed kernel's own emitted source. It does
// not modify inputs, operations, configuration or returned tensor values.
#include "original_tensor_io.h"
#include <cstdlib>

static bool observe_hyper_source() {
  const char *path=std::getenv("LUNA_ORIGINAL_HYPER_KERNEL_SOURCE");
  if(!path)return true;
  Owned module(PyImport_ImportModule("kernel"));
  Owned factory(module.p?PyObject_GetAttrString(module.p,"hc_split_sinkhorn_kernel"):nullptr);
  Owned compiled(factory.p?PyObject_CallFunction(factory.p,"iid",4,20,1e-6):nullptr);
  Owned source(compiled.p?PyObject_CallMethod(compiled.p,"get_kernel_source",nullptr):nullptr);
  if(!source.p)return false;
  Py_ssize_t length=0;const char *text=PyUnicode_AsUTF8AndSize(source.p,&length);
  if(!text||length<=0)return false;
  auto *file=std::fopen(path,"wbx");
  if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,path);return false;}
  const bool wrote=std::fwrite(text,1,length,file)==size_t(length);
  const bool closed=std::fclose(file)==0;
  return wrote&&closed;
}
