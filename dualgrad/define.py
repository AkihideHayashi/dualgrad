from typing import Callable, Tuple
from sympy import Symbol, symbols


def pythonize(name, func, inputs, parameters, simplify=False):
    inputs_x = tuple(Symbol("{}.func".format(inp)) for inp in inputs)
    f = func(*inputs_x, *parameters)
    if simplify:
        g = [f.diff(inp).simplify() for inp in inputs_x]
    else:
        g = [f.diff(inp) for inp in inputs_x]
    return """
def {}{}:
    return BackNumber({}, {}, {})
    """.format(name, (*inputs, *parameters), f, to_tuple(g), inputs)

def to_tuple(tpl):
    tpl = tuple(tpl)
    if len(tpl) > 1:
        return "({})".format(", ".join(map(str, tpl)))
    elif len(tpl) == 1:
        return "({},)".format(tpl[0])
    else:
        return "()"

def calc_diff_use_dual(function: Callable, inputs_real: Tuple[Symbol], inputs_dual: Tuple[Symbol], parameters: Tuple[Symbol]):
    func_real = function(*inputs_real, *parameters)
    grad_real = tuple(func_real.diff(i) for i in inputs_real)
    func_dual = sum(diff * dual for diff, dual in zip(grad_real, inputs_dual))
    grad_dual = tuple(sum(diff.diff(r) * d for r, d in zip(inputs_real, inputs_dual)) for diff in grad_real)
    return func_real, func_dual, grad_real, grad_dual

def cythonize(name: str, func: Callable, inputs: Tuple[Symbol], parameters: Tuple[Symbol]):
    inputs_real = tuple(Symbol('{}_real'.format(s.name)) for s in inputs)
    inputs_dual = tuple(Symbol('{}_dual'.format(s.name)) for s in inputs)
    func_real, func_dual, grad_real, grad_dual = calc_diff_use_dual(func, inputs_real, inputs_dual, parameters)
    
    cython_arguments = to_tuple("{}: double".format(p) for p in (*inputs_real, *inputs_dual, *parameters))
    cython = f"""cdef _{name}{cython_arguments}:
    return ({func_real}, {func_dual}), {grad_real}, {grad_dual}"""
    
    python_arguments = to_tuple(["{}: DualNumber".format(p) for p in (inputs)] + ["{}: float".format(p) for p in (parameters)])
    arg_real = ['{}.func.real'.format(p) for p in inputs]
    arg_dual = ['{}.func.dual'.format(p) for p in inputs]
    params = ['{}'.format(p) for p in parameters]
    arg_cython = to_tuple(arg_real + arg_dual + params)
    parents = to_tuple('{}'.format(p) for p in inputs)
    python = f"""def {name}{python_arguments}:
    func, grad_real, grad_dual = _{name}{arg_cython}
    return BackNumber(DualNumber(*func), tuple(DualNumber(r, g) for r, g in zip(grad_real, grad_dual)), {parents})"""
    
    return '\n\n'.join([python, cython])

def define_python(functions):
    imports = """from numpy import *
from dualgrad.special import *
from dualgrad import BackNumber, DualNumber"""
    return "\n\n".join([imports, *functions])

def define_cython(functions):
    imports = """from libc.math cimport *
from scipy.special.cython_special cimport *
from dualgrad import BackNumber, DualNumber"""
    return "\n\n".join([imports, *functions])