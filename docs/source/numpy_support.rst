==========================
 Numpy Support in *numba*
==========================

One objective of *numba* is having a seamless integration with *NumPy*. *NumPy* arrays provide
an efficient storage method for homogeneous sets if data. *NumPy* dtypes provide type information
useful when compiling, and the regular, structured storage of potentially large amounts of data
in memory provides an ideal memory layout for code generation. *numba* excels at generating code
that executes on top of *NumPy* arrays.

*NumPy* support in Numba comes in many forms:
* *numba* understands *NumPy* ufuncs and is able to generate equivalent native code for many of them.
* *NumPy* arrays are directly supported in *numba*. Access to *Numpy* arrays is very efficient, as indexing is lowered to memory accessing when possible.
* *numba* is able to generate ufuncs/gufuncs. This means that it is possible to implement ufuncs/gufuncs within Python, getting speeds comparable to that of ufuncs/gufuncs implemented in C extension modules using the NumPy C API.


Supported ufuncs
================

One objective on *numba* is having all the standard ufuncs in NumPy understood by numba. When a supported ufunc is found when compiling a function, *numba* maps the ufunc to equivalent native code. This allows the use of those ufuncs in *numba* code that gets compiled in no-python mode.

Limitations
-----------

Right now support for ufuncs is quite limited in no-python mode. Meaning that only a selection of the ufuncs work in no-python mode.

Also, in its current implementation ufuncs working on arrays will only compile in no-python mode if the output is explicit (the output array is passed as second argument). Note that this limitation does not apply when applied to scalars

Following is a list of the different *NumPy* ufuncs that *numba* is aware of, sorted in the same way as in the *NumPy* documentation.


Math operations
---------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 add                 Yes          Yes
 subtract            Yes          Yes
 multiply            Yes          Yes
 divide              Yes          Yes
 logaddexp           Yes          Yes
 logaddexp2          Yes          Yes
 true_divide         Yes          Yes
 floor_divide        Yes          Yes
 negative            Yes          Yes
 power               Yes          Yes
 remainder           Yes          Yes
 mod                 Yes          Yes
 fmod                Yes          Yes
 abs                 Yes          Yes
 absolute            Yes          Yes
 fabs                Yes          Yes
 rint                Yes          Yes
 sign                Yes          Yes
 conj                Yes          Yes
 exp                 Yes          Yes
 exp2                Yes          Yes
 log                 Yes          Yes
 log2                Yes          Yes
 log10               Yes          Yes
 expm1               Yes          Yes
 log1p               Yes          Yes
 sqrt                Yes          Yes
 square              Yes          Yes
 reciprocal          Yes          Yes
 conjugate           Yes          Yes
==============  =============  ===========


Trigonometric functions
-----------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 sin                 Yes          Yes
 cos                 Yes          Yes
 tan                 Yes          Yes
 arcsin              Yes          Yes
 arccos              Yes          Yes
 arctan              Yes          Yes
 arctan2             Yes          Yes
 hypot               Yes          Yes
 sinh                Yes          Yes
 cosh                Yes          Yes
 tanh                Yes          Yes
 arcsinh             Yes          Yes
 arccosh             Yes          Yes
 arctanh             Yes          Yes
 deg2rad             Yes          Yes
 rad2deg             Yes          Yes
 degrees             Yes          Yes
 radians             Yes          Yes
==============  =============  ===========


Bit-twiddling functions
-----------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 bitwise_and         Yes          Yes
 bitwise_or          Yes          Yes
 bitwise_xor         Yes          Yes
 bitwise_not         Yes          Yes
 invert              Yes          Yes
 left_shift          Yes          Yes
 right_shift         Yes          Yes
==============  =============  ===========


Comparison functions
--------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 greater             Yes          Yes
 greater_equal       Yes          Yes
 less                Yes          Yes
 less_equal          Yes          Yes
 not_equal           Yes          Yes
 equal               Yes          Yes
 logical_and         Yes          Yes
 logical_or          Yes          Yes
 logical_xor         Yes          Yes
 logical_not         Yes          Yes
 maximum             Yes          Yes
 minimum             Yes          Yes
 fmax                Yes          Yes
 fmin                Yes          Yes
==============  =============  ===========


Floating functions
------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 isfinite            Yes          Yes
 isinf               Yes          Yes
 isnan               Yes          Yes
 signbit             Yes          Yes
 copysign            Yes          Yes
 nextafter           Yes          Yes
 modf                Yes          No
 ldexp               Yes*         Yes
 frexp               Yes          No
 floor               Yes          Yes
 ceil                Yes          Yes
 trunc               Yes          Yes
 spacing             Yes          Yes
==============  =============  ===========

\* not supported on windows 32 bit
