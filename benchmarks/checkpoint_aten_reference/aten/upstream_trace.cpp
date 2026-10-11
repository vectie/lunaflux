// Offline observer of the UNMODIFIED checkpoint generate.py CLI.
// Capture its actual literal input/output lists, not re-decoded text or a new
// generator. This binary is never linked into LunaFlux's production runtime.
#include <Python.h>
#include <cstdio>
#include <cstring>
#include <string>

int install_upstream_stream_loader();

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *value) : p(value) {}
  ~Owned() { Py_XDECREF(p); }
  Owned(const Owned &) = delete;
};

static int emit(PyObject *input, PyObject *output) {
  if (!PyList_Check(input) || !PyList_Check(output) ||
      PyList_Size(input) != 1 || PyList_Size(output) != 1) {
    PyErr_SetString(PyExc_ValueError, "bounded upstream observer expects one sequence");
    return -1;
  }
  auto *prompt = PyList_GetItem(input, 0), *completion = PyList_GetItem(output, 0);
  if (!PyList_Check(prompt) || !PyList_Check(completion) ||
      PyList_Size(prompt) > 256 || PyList_Size(completion) > 9) {
    PyErr_SetString(PyExc_ValueError, "upstream observer extent exceeded");
    return -1;
  }
  Owned record(PyDict_New()), json(PyImport_ImportModule("json"));
  if (!record.p || !json.p ||
      PyDict_SetItemString(record.p, "prompt_tokens", input) < 0 ||
      PyDict_SetItemString(record.p, "completion_tokens", output) < 0) return -1;
  Owned encoded(PyObject_CallMethod(json.p, "dumps", "O", record.p));
  const char *text = encoded.p ? PyUnicode_AsUTF8(encoded.p) : nullptr;
  if (!text) return -1;
  std::printf("upstream_literal_tokens=%s\n", text);
  std::fflush(stdout);
  return 0;
}

static int trace(PyObject *filename, PyFrameObject *frame, int event, PyObject *result) {
  // A failing original function can emit a RETURN/None trace. Do not replace
  // its actual exception with an observer shape error.
  if (event != PyTrace_RETURN || result == Py_None) return 0;
  Owned code(reinterpret_cast<PyObject *>(PyFrame_GetCode(frame)));
  Owned file(code.p ? PyObject_GetAttrString(code.p, "co_filename") : nullptr);
  Owned name(code.p ? PyObject_GetAttrString(code.p, "co_name") : nullptr);
  if (!file.p || !name.p) return -1;
  const char *function = PyUnicode_AsUTF8(name.p);
  if (!function) return -1;
  if (std::strcmp(function, "generate") != 0) return 0;
  const int matched = PyObject_RichCompareBool(file.p, filename, Py_EQ);
  if (matched <= 0) return matched;
  Owned locals(PyFrame_GetLocals(frame));
  if (!locals.p) return -1;
  auto *input = PyDict_GetItemString(locals.p, "prompt_tokens");
  if (!input) { PyErr_SetString(PyExc_ValueError, "original generation input missing"); return -1; }
  return emit(input, result);
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fputs("ORIGINAL_GENERATE_SCRIPT [ORIGINAL_CLI_ARGS] | --probe\n", stderr); return 2; }
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.parse_argv = 0;
  auto status = PyConfig_SetBytesArgv(&config, argc - 1, argv + 1);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) { std::fputs("upstream observer interpreter initialization failed\n", stderr); return 1; }
  int failed = 0;
  {
    if (std::strcmp(argv[1], "--probe") == 0) {
      Owned input(Py_BuildValue("[[ii]]", 5, 223)), output(Py_BuildValue("[[ii]]", 19923, 1));
      failed = emit(input.p, output.p);
    } else {
      if(install_upstream_stream_loader()<0) failed=-1;
      std::string path(argv[1]);
      const auto slash = path.find_last_of('/');
      Owned directory(PyUnicode_FromString(path.substr(0, slash).c_str()));
      Owned filename(PyUnicode_FromString(path.c_str()));
      auto *paths = PySys_GetObject("path");
      if (!directory.p || !filename.p || !paths || PyList_Insert(paths, 0, directory.p) < 0) failed = -1;
      auto *file = failed ? nullptr : std::fopen(path.c_str(), "r");
      if (!file) { std::fputs("original CLI cannot be opened\n", stderr); failed = -1; }
      else {
        PyEval_SetTrace(trace, filename.p);
        failed = PyRun_SimpleFileExFlags(file, path.c_str(), 1, nullptr);
        PyEval_SetTrace(nullptr, nullptr);
      }
    }
    if (PyErr_Occurred()) { PyErr_Print(); failed = -1; }
  }
  if (Py_FinalizeEx() < 0) failed = -1;
  return failed ? 1 : 0;
}
