
//
// SVMHeavy python interface
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
//
// What is included:
//
// ML_Mutable: pretty much all of the ML blocks and associated functionality,
//             excluding those that require "non-trivial" classes like gentype,
//             SparseVector etc (but retaining basic function pointers, as used
//             eg in BLK_CalBak.
// addData:    basic loading and testing functionality only.
// errortest:  basic leave-one-out, cross-fold etc.
//
// Supports numpy for calls taking double *, int dim as arguments
// Callback for blk_calbak (208) also supports numpy
//
// Useful site: https://lindonjroberts.wordpress.com/2017/04/15/writing-c-extensions-for-python-with-numpy-and-callback-functions/
//

%module ml_mutable
%{
    #define SWIG_FILE_WITH_INIT
    #include "ml_mutable.h"
    #include "addData.h"
    #include "errortest.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double *xxa, int dima)};
%apply (double* IN_ARRAY1, int DIM1) {(double *xxb, int dimb)};

%apply (double** ARGOUTVIEW_ARRAY1, int* DIM1) {(double **res, int *dim)};

%include "ml_mutable.h"
%include "addData.h"
%include "errortest.h"

%include "carrays.i"
%array_functions(double,doubleArray);



