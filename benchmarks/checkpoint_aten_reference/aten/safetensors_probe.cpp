// Test the installed upstream serializer itself, independently of our writer.
#include <Python.h>
#include <cstdio>
#include <fstream>
#include <string>

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *v) : p(v) {}
  ~Owned() { Py_XDECREF(p); }
};

int install_upstream_stream_loader();

static int probe_loader(const char *path) {
  Owned nn(PyImport_ImportModule("torch.nn"));
  Owned safe(PyImport_ImportModule("safetensors.torch"));
  Owned torch(PyImport_ImportModule("torch"));
  Owned source_linear(nn.p?PyObject_CallMethod(nn.p,"Linear","ii",2,2):nullptr);
  Owned ordinary_linear(nn.p?PyObject_CallMethod(nn.p,"Linear","ii",2,2):nullptr);
  Owned streamed_linear(nn.p?PyObject_CallMethod(nn.p,"Linear","ii",2,2):nullptr);
  Owned source(source_linear.p?PyObject_CallMethod(nn.p,"Sequential","OO",source_linear.p,source_linear.p):nullptr);
  Owned ordinary(ordinary_linear.p?PyObject_CallMethod(nn.p,"Sequential","OO",ordinary_linear.p,ordinary_linear.p):nullptr);
  Owned streamed(streamed_linear.p?PyObject_CallMethod(nn.p,"Sequential","OO",streamed_linear.p,streamed_linear.p):nullptr);
  if(!source.p||!ordinary.p||!streamed.p||!safe.p||!torch.p) return -1;
  Owned saved(PyObject_CallMethod(safe.p,"save_model","Os",source.p,path));
  Owned baseline(saved.p?PyObject_CallMethod(safe.p,"load_model","Os",ordinary.p,path):nullptr);
  if(!baseline.p||install_upstream_stream_loader()<0) return -1;
  Owned changed(PyObject_CallMethod(safe.p,"load_model","Os",streamed.p,path));
  Owned first(PyObject_CallMethod(ordinary.p,"state_dict",nullptr));
  Owned second(PyObject_CallMethod(streamed.p,"state_dict",nullptr));
  if(!changed.p||!first.p||!second.p||PyObject_RichCompareBool(baseline.p,changed.p,Py_EQ)!=1) return -1;
  PyObject *key,*value; Py_ssize_t at=0;
  while(PyDict_Next(first.p,&at,&key,&value)) {
    auto *other=PyDict_GetItemWithError(second.p,key);
    Owned exact(other?PyObject_CallMethod(torch.p,"equal","OO",value,other):nullptr);
    if(!exact.p||PyObject_IsTrue(exact.p)!=1) return -1;
  }
  std::puts("stream_load_probe_passed shared_parameter_aliases=true upstream_result_exact=true tensor_values_exact=true");
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) return 2;
  Py_Initialize();
  int failed = 0;
  {
    Owned torch(PyImport_ImportModule("torch"));
    Owned safe(PyImport_ImportModule("safetensors.torch"));
    if (argc == 3 && std::string(argv[1]) == "--stream-load") {
      failed=probe_loader(argv[2]);
    } else if (argc == 3 && std::string(argv[1]) == "--read-first") {
      Owned module(PyImport_ImportModule("safetensors"));
      Owned handle(module.p ? PyObject_CallMethod(module.p,"safe_open","ss",argv[2],"pt") : nullptr);
      Owned keys(handle.p ? PyObject_CallMethod(handle.p,"keys",nullptr) : nullptr);
      auto *key=keys.p&&PyList_Size(keys.p)?PyList_GetItem(keys.p,0):nullptr;
      Owned value(key?PyObject_CallMethod(handle.p,"get_tensor","O",key):nullptr);
      Owned shape(value.p?PyObject_GetAttrString(value.p,"shape"):nullptr);
      if(!shape.p) failed=1;
      else std::printf("upstream_first_tensor_read key=%s type=%s\n",PyUnicode_AsUTF8(key),Py_TYPE(value.p)->tp_name);
    } else if (argc == 3 && std::string(argv[1]) == "--validate") {
      Owned module(PyImport_ImportModule("safetensors"));
      Owned handle(module.p ? PyObject_CallMethod(module.p, "safe_open", "ss", argv[2], "pt") : nullptr);
      Owned keys(handle.p ? PyObject_CallMethod(handle.p, "keys", nullptr) : nullptr);
      if (!keys.p) { PyErr_Print(); failed = 1; }
      else std::printf("upstream_reader_header_validated tensors=%zd\n", PyList_Size(keys.p));
    } else {
    Owned shape(Py_BuildValue("(ii)", 2, 4));
    Owned dtype(torch.p ? PyObject_GetAttrString(torch.p, "int8") : nullptr);
    Owned packed_dtype(torch.p ? PyObject_GetAttrString(torch.p, "float4_e2m1fn_x2") : nullptr);
    Owned full(torch.p ? PyObject_GetAttrString(torch.p, "full") : nullptr);
    Owned kwargs(PyDict_New());
    Owned args(Py_BuildValue("(Oi)", shape.p, 0x78));
    if (!torch.p || !safe.p || !dtype.p || !packed_dtype.p || !full.p ||
        !kwargs.p || !args.p || PyDict_SetItemString(kwargs.p, "dtype", dtype.p) < 0) failed = 1;
    Owned bytes(!failed ? PyObject_Call(full.p, args.p, kwargs.p) : nullptr);
    Owned packed(bytes.p ? PyObject_CallMethod(bytes.p, "view", "O", packed_dtype.p) : nullptr);
    Owned tensors(PyDict_New());
    if (!packed.p || !tensors.p || PyDict_SetItemString(tensors.p, "packed", packed.p) < 0) failed = 1;
    Owned saved(!failed ? PyObject_CallMethod(safe.p, "save_file", "Os", tensors.p, argv[1]) : nullptr);
    if (!saved.p) failed = 1;
    if (!failed) {
      std::ifstream f(argv[1], std::ios::binary);
      unsigned char prefix[8]; f.read(reinterpret_cast<char *>(prefix), 8);
      size_t n = 0; for (unsigned i = 0; i < 8; ++i) n |= size_t(prefix[i]) << (i * 8);
      if (!f || n > 4096) failed = 1;
      else { std::string header(n, ' '); f.read(header.data(), n); std::printf("upstream_packed_header=%s\n", header.c_str()); }
      Owned loaded(PyObject_CallMethod(safe.p, "load_file", "s", argv[1]));
      auto *tensor = loaded.p ? PyDict_GetItemString(loaded.p, "packed") : nullptr;
      // Compare dtype and shape, then original physical bytes through int8.
      Owned tensor_shape(tensor ? PyObject_GetAttrString(tensor, "shape") : nullptr);
      Owned original_shape(PyObject_GetAttrString(packed.p, "shape"));
      Owned tensor_dtype(tensor ? PyObject_GetAttrString(tensor, "dtype") : nullptr);
      Owned restored(tensor ? PyObject_CallMethod(tensor, "view", "O", dtype.p) : nullptr);
      Owned exact(restored.p ? PyObject_CallMethod(torch.p, "equal", "OO", restored.p, bytes.p) : nullptr);
      if (!tensor_shape.p || !original_shape.p || !tensor_dtype.p || !exact.p ||
          PyObject_RichCompareBool(tensor_shape.p, original_shape.p, Py_EQ) != 1 ||
          PyObject_RichCompareBool(tensor_dtype.p, packed_dtype.p, Py_EQ) != 1 ||
          PyObject_IsTrue(exact.p) != 1) failed = 1;
    }
    }
    if (PyErr_Occurred()) { PyErr_Print(); failed = 1; }
  }
  if (Py_FinalizeEx() < 0) failed = 1;
  return failed;
}
