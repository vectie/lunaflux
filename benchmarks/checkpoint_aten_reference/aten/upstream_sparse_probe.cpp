// Offline launch-capacity probe of the original DeepSeek sparse attention.
// No production linkage and no replacement attention arithmetic.
#include <Python.h>
#include <cstdio>

struct Ref {
  PyObject *p;
  explicit Ref(PyObject *v) : p(v) {}
  ~Ref() { Py_XDECREF(p); }
  Ref(const Ref &) = delete;
};

static PyObject *tensor(PyObject *torch, const char *factory, PyObject *shape,
                        const char *dtype) {
  Ref fn(PyObject_GetAttrString(torch, factory));
  Ref args(PyTuple_Pack(1, shape)), kw(PyDict_New());
  Ref type(PyObject_GetAttrString(torch, dtype)), device(PyUnicode_FromString("cuda"));
  if (!fn.p || !args.p || !kw.p || !type.p || !device.p ||
      PyDict_SetItemString(kw.p, "dtype", type.p) < 0 ||
      PyDict_SetItemString(kw.p, "device", device.p) < 0) return nullptr;
  return PyObject_Call(fn.p, args.p, kw.p);
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  Py_Initialize();
  int failed = 0;
  {
    Ref directory(PyUnicode_FromString(argv[1]));
    if (!directory.p || PyList_Insert(PySys_GetObject("path"), 0, directory.p) < 0) failed = 1;
    Ref torch(failed ? nullptr : PyImport_ImportModule("torch"));
    Ref cuda(torch.p ? PyObject_GetAttrString(torch.p, "cuda") : nullptr);
    Ref properties(cuda.p ? PyObject_CallMethod(cuda.p, "get_device_properties", "i", 0) : nullptr);
    Ref capacity(properties.p ? PyObject_GetAttrString(properties.p, "shared_memory_per_block_optin") : nullptr);
    if (capacity.p) std::printf("device_shared_optin_bytes=%ld\n", PyLong_AsLong(capacity.p));
    Ref kernel(torch.p ? PyImport_ImportModule("kernel") : nullptr);
    Ref qs(Py_BuildValue("(iiii)", 1, 1, 32, 512));
    Ref ks(Py_BuildValue("(iii)", 1, 128, 512));
    Ref ss(Py_BuildValue("(i)", 32));
    Ref is(Py_BuildValue("(iii)", 1, 1, 128));
    Ref q(torch.p ? tensor(torch.p, "zeros", qs.p, "bfloat16") : nullptr);
    Ref kv(torch.p ? tensor(torch.p, "ones", ks.p, "bfloat16") : nullptr);
    Ref sink(torch.p ? tensor(torch.p, "zeros", ss.p, "float32") : nullptr);
    Ref ids(torch.p ? tensor(torch.p, "zeros", is.p, "int32") : nullptr);
    Ref output(kernel.p && q.p && kv.p && sink.p && ids.p ?
      PyObject_CallMethod(kernel.p, "sparse_attn", "OOOOd", q.p, kv.p, sink.p, ids.p, 0.04419417382415922) : nullptr);
    Ref sync(output.p ? PyObject_CallMethod(cuda.p, "synchronize", nullptr) : nullptr);
    Ref finite(output.p ? PyObject_CallMethod(torch.p, "isfinite", "O", output.p) : nullptr);
    Ref all(finite.p ? PyObject_CallMethod(finite.p, "all", nullptr) : nullptr);
    Ref item(all.p ? PyObject_CallMethod(all.p, "item", nullptr) : nullptr);
    Ref mean(output.p ? PyObject_CallMethod(output.p, "mean", nullptr) : nullptr);
    Ref value(mean.p ? PyObject_CallMethod(mean.p, "item", nullptr) : nullptr);
    if (!sync.p || !item.p || !value.p || PyObject_IsTrue(item.p) != 1) failed = 1;
    else std::printf("original_sparse_launch=true finite=true output_mean=%.9g\n", PyFloat_AsDouble(value.p));
    if (PyErr_Occurred()) { PyErr_Print(); failed = 1; }
  }
  if (Py_FinalizeEx() < 0) failed = 1;
  return failed;
}
