// Offline Pillow transport/fixture adapter. Orientation/RGB reference semantics
// execute the unchanged pinned Transformers load_image and Pillow methods.
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
    if (!p) { PyErr_Print(); throw std::runtime_error("image reference call failed"); }
  }
  ~Object() { Py_DECREF(p); }
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
};
static PyObject* requires(PyObject*, PyObject*) { Py_RETURN_NONE; }
static PyMethodDef require_hook = {"requires_backends", requires, METH_VARARGS, nullptr};

static Object bytes_file(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("image input missing");
  std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  return Object(PyBytes_FromStringAndSize(bytes.data(), bytes.size()));
}
static void write_bytes(PyObject* image, const std::string& path) {
  Object bytes(PyObject_CallMethod(image, "tobytes", nullptr));
  std::ofstream output(path, std::ios::binary);
  output.write(PyBytes_AsString(bytes.p), PyBytes_Size(bytes.p));
  if (!output) throw std::runtime_error("image serialization failed");
}
static Object exif(int orientation, bool little) {
  const unsigned char le[] = {'E','x','i','f',0,0,'I','I',42,0,8,0,0,0,1,0,18,1,3,0,1,0,0,0,0,0,0,0,0,0,0,0};
  const unsigned char be[] = {'E','x','i','f',0,0,'M','M',0,42,0,0,0,8,0,1,1,18,0,3,0,0,0,1,0,0,0,0,0,0,0,0};
  std::string payload(reinterpret_cast<const char*>(little ? le : be), sizeof(le));
  payload[little ? 24 : 25] = static_cast<char>(orientation);
  return Object(PyBytes_FromStringAndSize(payload.data(), payload.size()));
}
static void metadata(PyObject* image, PyObject* pil, const std::string& prefix) {
  Object json(PyImport_ImportModule("json"));
  Object width(PyObject_GetAttrString(image, "width"));
  Object height(PyObject_GetAttrString(image, "height"));
  Object version(PyObject_GetAttrString(pil, "__version__"));
  Object value(Py_BuildValue("{s:O,s:O,s:O}", "width", width.p, "height", height.p, "pillow_version", version.p));
  Object encoded(PyObject_CallMethod(json.p, "dumps", "O", value.p));
  std::ofstream output(prefix + ".json");
  output << PyUnicode_AsUTF8(encoded.p) << '\n';
  if (!output) throw std::runtime_error("metadata serialization failed");
}
static int run(int argc, char** argv) {
  Object module(PyModule_New("image_load_reference"));
  PyObject* globals = PyModule_GetDict(module.p);
  Object require_fn(PyCFunction_New(&require_hook, nullptr));
  if (PyDict_SetItemString(globals, "requires_backends", require_fn.p) != 0)
    throw std::runtime_error("reference hook failed");
  FILE* source = std::fopen(argv[2], "r");
  if (!source) throw std::runtime_error("upstream loader source missing");
  Object loaded(PyRun_FileEx(source, argv[2], Py_file_input, globals, globals, 1));
  PyObject* loader = PyDict_GetItemString(globals, "load_image");
  if (!loader) throw std::runtime_error("upstream loader missing");
  Object pil(PyImport_ImportModule("PIL"));
  Object image_module(PyImport_ImportModule("PIL.Image"));
  const bool make = std::string(argv[3]) == "make";
  if ((make && argc != 13) || (!make && argc != 10)) throw std::runtime_error("invalid fixture arguments");
  const int width = std::stoi(argv[make ? 6 : 4]);
  const int height = std::stoi(argv[make ? 7 : 5]);
  const int orientation = std::stoi(argv[make ? 8 : 6]);
  const bool little = std::string(argv[make ? 9 : 7]) == "little";
  Object input(bytes_file(argv[make ? 11 : 8]));
  Object image(PyObject_CallMethod(image_module.p, "frombytes", "s(ii)O", make ? argv[4] : "RGB", width, height, input.p));
  Object orientation_bytes(exif(orientation, little));
  const std::string prefix = argv[make ? 12 : 9];
  if (make) {
    Object path(PyUnicode_FromString((prefix + ".encoded").c_str()));
    Object args(PyTuple_Pack(1, path.p));
    Object kwargs(PyDict_New());
    Object format(PyUnicode_FromString(argv[5]));
    if (PyDict_SetItemString(kwargs.p, "format", format.p) != 0) throw std::runtime_error("format failed");
    if (orientation != 0 && PyDict_SetItemString(kwargs.p, "exif", orientation_bytes.p) != 0)
      throw std::runtime_error("EXIF argument failed");
    if (std::string(argv[5]) == "JPEG") {
      Object quality(PyLong_FromLong(95));
      Object subsampling(PyLong_FromLong(0));
      PyDict_SetItemString(kwargs.p, "quality", quality.p);
      PyDict_SetItemString(kwargs.p, "subsampling", subsampling.p);
      PyDict_SetItemString(kwargs.p, "progressive", std::string(argv[10]) == "true" ? Py_True : Py_False);
    }
    Object save(PyObject_GetAttrString(image.p, "save"));
    Object saved(PyObject_Call(save.p, args.p, kwargs.p));
    Object stored(PyObject_CallMethod(image_module.p, "open", "O", path.p));
    Object stored_rgb(PyObject_CallMethod(stored.p, "convert", "s", "RGB"));
    write_bytes(stored_rgb.p, prefix + ".stored.rgb");
    Object oriented(PyObject_CallOneArg(loader, path.p));
    write_bytes(oriented.p, prefix + ".rgb");
    metadata(oriented.p, pil.p, prefix);
  } else {
    Object info(PyObject_GetAttrString(image.p, "info"));
    if (orientation != 0 && PyDict_SetItemString(info.p, "exif", orientation_bytes.p) != 0)
      throw std::runtime_error("native RGB metadata assignment failed");
    Object oriented(PyObject_CallOneArg(loader, image.p));
    write_bytes(oriented.p, prefix + ".rgb");
    metadata(oriented.p, pil.p, prefix);
  }
  return 0;
}
int main(int argc, char** argv) {
  if (argc < 4) return 2;
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, argv[1]);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) Py_ExitStatusException(status);
  int result = 1;
  try { result = run(argc, argv); }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
  Py_FinalizeEx();
  return result;
}
