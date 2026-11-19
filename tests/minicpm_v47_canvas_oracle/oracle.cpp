// Offline adapter only. Position arithmetic/grouping executes the unchanged
// pinned upstream methods; no model weights or CUDA are loaded.
#include <Python.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

struct Object {
  PyObject* p;
  explicit Object(PyObject* value) : p(value) {
    if (!p) { PyErr_Print(); throw std::runtime_error("Python reference call failed"); }
  }
  ~Object() { Py_DECREF(p); }
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
};

static std::string read(const char* path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("reference input unavailable");
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static PyObject* parse(PyObject* json, const char* path) {
  const auto text = read(path);
  Object input(PyUnicode_FromStringAndSize(text.data(), text.size()));
  return PyObject_CallMethod(json, "loads", "O", input.p);
}

static PyObject* tensor(PyObject* torch, PyObject* data) {
  Object factory(PyObject_GetAttrString(torch, "tensor"));
  Object args(PyTuple_Pack(1, data));
  Object dtype(PyObject_GetAttrString(torch, "long"));
  Object cpu(PyUnicode_FromString("cpu"));
  Object keywords(PyDict_New());
  PyDict_SetItemString(keywords.p, "dtype", dtype.p);
  PyDict_SetItemString(keywords.p, "device", cpu.p);
  return PyObject_Call(factory.p, args.p, keywords.p);
}

static int run(char** argv) {
  Object torch(PyImport_ImportModule("torch"));
  Object json(PyImport_ImportModule("json"));
  Object types(PyImport_ImportModule("types"));
  Object config(parse(json.p, argv[3]));
  Object corpus(parse(json.p, argv[4]));
  Object globals(PyDict_New());
  PyDict_SetItemString(globals.p, "__builtins__", PyEval_GetBuiltins());
  FILE* source = std::fopen(argv[2], "r");
  if (!source) throw std::runtime_error("reference source unavailable");
  Object evaluated(PyRun_FileEx(source, argv[2], Py_file_input, globals.p, globals.p, 1));
  PyObject* klass = PyDict_GetItemString(globals.p, "CanvasReference");
  if (!klass) throw std::runtime_error("reference class missing");
  Object instance(PyObject_CallNoArgs(klass));
  Object namespace_type(PyObject_GetAttrString(types.p, "SimpleNamespace"));
  Object empty_args(PyTuple_New(0));
  Object namespace_config(PyObject_Call(namespace_type.p, empty_args.p, config.p));
  if (PyObject_SetAttrString(instance.p, "config", namespace_config.p) != 0)
    throw std::runtime_error("reference configuration failed");
  std::ofstream output(argv[5]);
  if (!output) throw std::runtime_error("reference output unavailable");
  Object version(PyObject_GetAttrString(torch.p, "__version__"));
  Object header(PyDict_New());
  PyDict_SetItemString(header.p, "torch_version", version.p);
  Object encoded_header(PyObject_CallMethod(json.p, "dumps", "O", header.p));
  output << PyUnicode_AsUTF8(encoded_header.p) << '\n';
  for (Py_ssize_t i = 0; i < PyList_Size(corpus.p); ++i) {
    PyObject* item = PyList_GetItem(corpus.p, i);
    Object tokens(tensor(torch.p, PyDict_GetItemString(item, "tokens")));
    Object modalities(tensor(torch.p, PyDict_GetItemString(item, "modalities")));
    Object images(tensor(torch.p, PyDict_GetItemString(item, "image_grids")));
    Object videos(tensor(torch.p, PyDict_GetItemString(item, "video_grids")));
    Object keywords(PyDict_New());
    PyDict_SetItemString(keywords.p, "input_ids", tokens.p);
    PyDict_SetItemString(keywords.p, "mm_token_type_ids", modalities.p);
    PyDict_SetItemString(keywords.p, "target_sizes", images.p);
    PyDict_SetItemString(keywords.p, "target_sizes_videos", videos.p);
    PyDict_SetItemString(keywords.p, "downsample_mode", PyDict_GetItemString(item, "downsample"));
    Object method(PyObject_GetAttrString(instance.p, "get_rope_index"));
    Object result(PyObject_Call(method.p, empty_args.p, keywords.p));
    Object positions(PyObject_CallMethod(PyTuple_GetItem(result.p, 0), "tolist", nullptr));
    Object delta(PyObject_CallMethod(PyTuple_GetItem(result.p, 1), "tolist", nullptr));
    Object record(PyDict_New());
    PyDict_SetItemString(record.p, "name", PyDict_GetItemString(item, "name"));
    PyDict_SetItemString(record.p, "positions", positions.p);
    PyDict_SetItemString(record.p, "rope_delta", delta.p);
    Object encoded(PyObject_CallMethod(json.p, "dumps", "O", record.p));
    output << PyUnicode_AsUTF8(encoded.p) << '\n';
  }
  return output ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 6) throw std::runtime_error("PYTHON_EXECUTABLE REFERENCE_SOURCE CONFIG CORPUS OUTPUT");
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
