
#include "iostream"
#include "Python.h"

int callme(int x);
int callme2(double x[4]);

//PyObject *my_set_callback(PyObject *args);

double PythonCallBack(double a, int b, PyObject *clientdata);
double PythonCallBackB(double a, int b, PyObject *clientdata);


