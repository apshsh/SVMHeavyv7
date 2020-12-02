
%module testit
%{
    #include "testit.h"
%}

%include "numpy.i"

// Grab a 4 element array as a Python 4-tuple
%typemap(in) double[4](double temp[4]) {   // temp[4] becomes a local variable
  int i;
  if (PyTuple_Check($input)) {
    if (!PyArg_ParseTuple($input, "dddd", temp, temp+1, temp+2, temp+3)) {
      PyErr_SetString(PyExc_TypeError, "tuple must have 4 elements");
      SWIG_fail;
    }
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError, "expected a tuple.");
    SWIG_fail;
  }
}

// Grab a Python function object as a Python object.
%typemap(in) PyObject *pyfunc {
  if (!PyCallable_Check($input)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object!");
      return NULL;
  }
  $1 = $input;
}


%include "testit.h"



