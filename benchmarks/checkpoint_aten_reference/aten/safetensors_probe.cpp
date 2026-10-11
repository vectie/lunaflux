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

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) return 2;
  Py_Initialize();
  int failed = 0;
  {
    Owned torch(PyImport_ImportModule("torch"));
    Owned safe(PyImport_ImportModule("safetensors.torch"));
    if (argc == 3 && std::string(argv[1]) == "--validate") {
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
