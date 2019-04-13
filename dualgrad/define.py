import sympy

def to_tuple(tpl):
    if len(tpl) > 1:
        return "({})".format(", ".join(map(str, tpl)))
    else:
        return "({},)".format(tpl[0])

def define_function(name, func, inputs, parameters, simplify=False):
    inputs_x = sympy.symbols(tuple("{}.x".format(inp) for inp in inputs))
    inputs = sympy.symbols(inputs)
    parameters = sympy.symbols(parameters)
    f = func(*inputs_x, *parameters)
    if simplify:
        g = [sympy.simplify(sympy.diff(f, inp)) for inp in inputs_x]
    else:
        g = [sympy.diff(f, inp) for inp in inputs_x]
    return """
def {}{}:
    for node in {}:
        node.u += 1
    return BackNumber({}, {}, {})
    """.format(name, (*inputs, *parameters), inputs, f, inputs, to_tuple(g))

def define_functions(functions):
    imports = """from dualgrad.special import *
from dualgrad import BackNumber, DualNumber"""
    return "\n\n".join([imports, *functions])