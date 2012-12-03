import numpy as np

import llvm.core
import llvm.ee

from numba import decorators, ast_translate
from numba.minivect import minitypes
from llvm_cbuilder import shortnames as _C
from numba.vectorize import _internal
from ._translate import Translate
from llvm.passes import PassManager, PassManagerBuilder

try:
    ptr_t = long
except:
    ptr_t = int
    assert False, "Have not check this yet" # Py3.0?

_llvm_ty_str_to_numpy = {
            'i8'     : np.int8,
            'i16'    : np.int16,
            'i32'    : np.int32,
            'i64'    : np.int64,
            'float'  : np.float32,
            'double' : np.float64,
        }

def _llvm_ty_to_numpy(ty):
    return _llvm_ty_str_to_numpy[str(ty)]

def _llvm_ty_to_dtype(ty):
    return np.dtype(_llvm_ty_to_numpy(ty)) #.num

_numbatypes_str_to_numpy = {
            'int8'     : np.int8,
            'int16'    : np.int16,
            'int32'    : np.int32,
            'int64'    : np.int64,
            'uint8'    : np.uint8,
            'uint16'   : np.uint16,
            'uint32'   : np.uint32,
            'uint64'   : np.uint64,
#            'f'        : np.float32,
#            'd'        : np.float64,
            'float'    : np.float32,
            'double'   : np.float64,
        }

def _numbatypes_to_numpy(ty):
    ret = _numbatypes_str_to_numpy[str(ty)]
    return ret

class CommonVectorizeFromFunc(object):
    def build(self, lfunc, dtypes):
        raise NotImplementedError

    def get_dtype_nums(self, tyslist):
        return [[dtype.num for dtype in dtypes] for dtypes in tyslist]

    def __call__(self, lfunclist, tyslist, engine,
                 dispatcher=None,
                 **kws):
        '''create ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : a llvm.ee.ExecutionEngine instance

        return a function object which can be called from python.
        '''
        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]

        ptrlist = self._prepare_pointers(lfunclist, tyslist, engine, **kws)

        fntype = lfunclist[0].type.pointee
        inct = len(fntype.args)
        outct = 1

        datlist = [None] * len(lfunclist)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        tyslist = self.get_dtype_nums(tyslist)
        ufunc = _internal.fromfunc(ptrlist, tyslist, inct, outct,
                                   datlist, dispatcher)
        return ufunc

    def _prepare_ufunc_core(self, lfunclist, tyslist, **kws):
        spuflist = []
        for i, (lfunc, dtypes) in enumerate(zip(lfunclist, tyslist)):
            spuflist.append(self.build(lfunc, dtypes, **kws))
        return spuflist

    def _get_pointer_from_ufunc_core(self, spuf, engine):
        fptr = engine.get_pointer_to_function(spuf)
        return ptr_t(fptr)

    def _prepare_pointers(self, lfunclist, tyslist, engine, **kws):
        spuflist = self._prepare_ufunc_core(lfunclist, tyslist, **kws)
        ptrlist = [self._get_pointer_from_ufunc_core(spuf, engine)
                   for spuf in spuflist]
        return ptrlist

    def _prepare_prototypes_and_pointers(self, lfunclist, tyslist, engine, **kws):
        spuflist = self._prepare_ufunc_core(lfunclist, tyslist, **kws)
        ptrlist = [self._get_pointer_from_ufunc_core(spuf, engine)
                   for spuf in spuflist]
        return zip(spuflist, ptrlist)

class GenericVectorize(object):
    def __init__(self, func):
        self.pyfunc = func
        self.translates = []
        self.args_restypes = []

    def add(self, *args, **kwargs):
        t = Translate(self.pyfunc, *args, **kwargs)
        t.translate()
        self.translates.append(t)

        argtys = kwargs['argtypes']
        retty = kwargs['restype']
        self.args_restypes.append(argtys + [retty])

    def _get_tys_list(self):
        tyslist = []
        for args_ret in self.args_restypes:
            tys = []
            for ty in args_ret:
                tys.append(np.dtype(_numbatypes_to_numpy(ty)))
            tyslist.append(tys)
        return tyslist

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def _get_ee(self):
        return self.translates[0]._get_ee()

    def build_ufunc(self):
        raise NotImplementedError

    def _from_func(self, **kws):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self._get_ee()
        return self._from_func_factory(lfunclist, tyslist, engine=engine, **kws)


    def build_ufunc_core(self, **kws):
        '''Build the ufunc core functions and returns the prototype and pointer.
        The return value is a list of tuples (prototype, pointer).
        '''
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self._get_ee()
        get_proto_ptr = self._from_func_factory._prepare_prototypes_and_pointers
        return get_proto_ptr(lfunclist, tyslist, engine, **kws)

class ASTVectorizeMixin(object):

    def __init__(self, *args, **kwargs):
        super(ASTVectorizeMixin, self).__init__(*args, **kwargs)
        self.llvm_context = ast_translate.LLVMContextManager()
        self.mod = self.llvm_context.get_default_module()
        self.ee = self.llvm_context.get_execution_engine()
        self.args_restypes = getattr(self, 'args_restypes', [])
        self.signatures = []

    def _get_ee(self):
        return self.ee

    def add(self, restype=None, argtypes=None):
        dec = decorators.jit2(restype, argtypes,
                              _llvm_module=self.mod,
                              _llvm_ee=self.ee)
        numba_func = dec(self.pyfunc)
        self.args_restypes.append(list(numba_func.signature.args) +
                                   [numba_func.signature.return_type])
        self.signatures.append((restype, argtypes, {}))
        self.translates.append(numba_func)

    def get_argtypes(self, numba_func):
        return list(numba_func.signature.args) + [numba_func.signature.return_type]

    def _get_tys_list(self):
        types_lists = []
        for numba_func in self.translates:
            dtype_nums = []
            types_lists.append(dtype_nums)
            for arg_type in self.get_argtypes(numba_func):
                dtype = minitypes.map_minitype_to_dtype(arg_type)
                dtype_nums.append(dtype)

        return types_lists

class GenericASTVectorize(ASTVectorizeMixin, GenericVectorize):
    "Use the AST backend to compile the ufunc"

def post_vectorize_optimize(func):
    '''Perform aggressive optimization after each vectorizer.

    TODO: Currently uses Module level PassManager each is rather wasteful
          and may have side-effect on other already optimized functions.
          We should find out a list of optimization to add use in
          FunctionPassManager.
    '''
    pmb = PassManagerBuilder.new()
    pmb.opt_level = 3
    pmb.vectorize = True

    pm = PassManager.new()
    pmb.populate(pm)

    pm.run(func.module)