// Offline transport/serialization adapter. All preprocessing arithmetic,
// geometry and patch ordering execute unchanged pinned upstream methods.
#include <Python.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

struct Object {
  PyObject* p;
  explicit Object(PyObject* value) : p(value) {
    if (!p) { PyErr_Print(); throw std::runtime_error("PIL reference call failed"); }
  }
  ~Object() { Py_DECREF(p); }
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
};

// Dependency probes and BatchFeature's unused tensor-conversion wrapper do
// not participate in arithmetic. The reference returns its original data dict.
static PyObject* available(PyObject*, PyObject*) { Py_RETURN_TRUE; }
static PyObject* requires(PyObject*, PyObject*) { Py_RETURN_NONE; }
static PyObject* is_tensor(PyObject*, PyObject* value) {
  Object torch(PyImport_ImportModule("torch"));
  Object type(PyObject_GetAttrString(torch.p, "Tensor"));
  const int result = PyObject_IsInstance(value, type.p);
  if (result < 0) return nullptr;
  return PyBool_FromLong(result);
}
static PyObject* batch_feature(PyObject*, PyObject*, PyObject* kwargs) {
  PyObject* data = PyDict_GetItemString(kwargs, "data");
  if (!data) { PyErr_SetString(PyExc_ValueError, "BatchFeature data missing"); return nullptr; }
  return Py_NewRef(data);
}
static PyMethodDef hooks[] = {
  {"is_torch_available", available, METH_NOARGS, nullptr},
  {"requires_backends", requires, METH_VARARGS, nullptr},
  {"is_torch_tensor", is_tensor, METH_O, nullptr},
  {"BatchFeature", _PyCFunction_CAST(batch_feature), METH_VARARGS | METH_KEYWORDS, nullptr},
};

static void set(PyObject* dict, const char* key, PyObject* value) {
  if (PyDict_SetItemString(dict, key, value) != 0)
    throw std::runtime_error("reference argument assignment failed");
}

static int run(char** argv) {
  // dataclasses resolves annotations via the module registry.
  Object module(PyModule_New("pil_reference"));
  set(PyImport_GetModuleDict(), "pil_reference", module.p);
  PyObject* module_dict = PyModule_GetDict(module.p);
  for (auto& hook : hooks) {
    Object function(PyCFunction_New(&hook, nullptr));
    set(module_dict, hook.ml_name, function.p);
  }
  FILE* source = std::fopen(argv[2], "r");
  if (!source) throw std::runtime_error("reference source unavailable");
  Object evaluated(PyRun_FileEx(source, argv[2], Py_file_input, module_dict, module_dict, 1));
  PyObject* klass = PyDict_GetItemString(module_dict, "ProcessorReference");
  if (!klass) throw std::runtime_error("reference class missing");
  Object instance(PyObject_CallNoArgs(klass));
  Object numpy(PyImport_ImportModule("numpy"));
  Object torch(PyImport_ImportModule("torch"));
  Object json(PyImport_ImportModule("json"));
  Object pil(PyImport_ImportModule("PIL"));
  const int height = std::stoi(argv[4]);
  const int width = std::stoi(argv[5]);
  std::ifstream input(argv[3], std::ios::binary);
  if (!input) throw std::runtime_error("RGB input unavailable");
  const std::string data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  if (data.size() != static_cast<size_t>(height) * width * 3)
    throw std::runtime_error("RGB input length differs");
  Object bytes(PyBytes_FromStringAndSize(data.data(), data.size()));
  Object uint8(PyObject_GetAttrString(numpy.p, "uint8"));
  Object flat(PyObject_CallMethod(numpy.p, "frombuffer", "OO", bytes.p, uint8.p));
  Object hwc(PyObject_CallMethod(flat.p, "reshape", "iii", height, width, 3));
  Object chw(PyObject_CallMethod(hwc.p, "transpose", "iii", 2, 0, 1));
  Object images(PyList_New(1));
  PyList_SET_ITEM(images.p, 0, Py_NewRef(chw.p));
  Object kwargs(PyDict_New());
  set(kwargs.p, "images", images.p);
  set(kwargs.p, "do_resize", Py_True);
  Object resample(PyLong_FromLong(3)); // Pillow BICUBIC
  set(kwargs.p, "resample", resample.p);
  set(kwargs.p, "do_rescale", Py_True);
  Object factor(PyFloat_FromDouble(1.0 / 255.0));
  set(kwargs.p, "rescale_factor", factor.p);
  set(kwargs.p, "do_normalize", Py_True);
  Object means(Py_BuildValue("[ddd]", 0.5, 0.5, 0.5));
  set(kwargs.p, "image_mean", means.p);
  set(kwargs.p, "image_std", means.p);
  Object slices(PyLong_FromLong(std::stoi(argv[6])));
  Object resolution(PyLong_FromLong(448));
  Object patch(PyLong_FromLong(14));
  set(kwargs.p, "max_slice_nums", slices.p);
  set(kwargs.p, "scale_resolution", resolution.p);
  set(kwargs.p, "patch_size", patch.p);
  set(kwargs.p, "slice_mode", Py_True);
  Object method(PyObject_GetAttrString(instance.p, "_preprocess"));
  Object args(PyTuple_New(0));
  Object result(PyObject_Call(method.p, args.p, kwargs.p));
  PyObject* pixels = PyDict_GetItemString(result.p, "pixel_values");
  Object tensor(PyObject_CallMethod(torch.p, "from_numpy", "O", pixels));
  Object bf16(PyObject_GetAttrString(torch.p, "bfloat16"));
  Object cast(PyObject_CallMethod(tensor.p, "to", "O", bf16.p));
  Object int16(PyObject_GetAttrString(torch.p, "int16"));
  Object words(PyObject_CallMethod(cast.p, "view", "O", int16.p));
  Object array(PyObject_CallMethod(words.p, "numpy", nullptr));
  Object little_endian(PyObject_CallMethod(array.p, "astype", "s", "<u2"));
  Object payload(PyObject_CallMethod(little_endian.p, "tobytes", nullptr));
  std::ofstream binary(argv[7], std::ios::binary);
  binary.write(PyBytes_AsString(payload.p), PyBytes_Size(payload.p));
  if (!binary) throw std::runtime_error("BF16 serialization failed");
  Object metadata(PyDict_New());
  Object shape(PyObject_GetAttrString(pixels, "shape"));
  Object sizes(PyObject_CallMethod(PyDict_GetItemString(result.p, "target_sizes"), "tolist", nullptr));
  set(metadata.p, "shape", shape.p);
  set(metadata.p, "target_sizes", sizes.p);
  set(metadata.p, "grids", PyDict_GetItemString(result.p, "grids"));
  set(metadata.p, "num_patches_per_image", PyDict_GetItemString(result.p, "num_patches_per_image"));
  for (auto pair : {std::pair<const char*, PyObject*>{"torch_version", torch.p},
                    {"numpy_version", numpy.p}, {"pillow_version", pil.p}}) {
    Object version(PyObject_GetAttrString(pair.second, "__version__"));
    set(metadata.p, pair.first, version.p);
  }
  Object encoded(PyObject_CallMethod(json.p, "dumps", "O", metadata.p));
  std::ofstream output(argv[8]);
  output << PyUnicode_AsUTF8(encoded.p) << '\n';
  return output ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 9) throw std::runtime_error("PYTHON REFERENCE RGB H W MAX_SLICES BF16 METADATA");
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, argv[1]);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) Py_ExitStatusException(status);
  int result = 1;
  try { result = run(argv); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
  Py_FinalizeEx();
  return result;
}
