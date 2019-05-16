import numpy as np
from collections import deque
import sympy
from .dual import DualNumber

def _define_binary(f, g, x, y):
    if isinstance(y, BackNumber):
        return BackNumber(f(x.func, y.func), (g(x.func, y.func)), (x, y))
    else:
        return BackNumber(f(x.func, y), (g(x.func, y)[0],), (x,))

class BackNumber(object):
    """
    func: the value of this variable
    grad: the gradient of this variable
    p: the parent nodes
    c: differential coeffients
    u: how many times this variable used
    """
    def __init__(self, func, coefficients=(), parents=()):
        for parent in parents:
            parent.u += 1
        self.func = func      # value
        if isinstance(func, DualNumber):
            self.grad = DualNumber(0.0, 0.0)
        else:
            self.grad = 0.0       # grad
        self.p = parents      # parent_nodes
        self.c = coefficients # differential coefficients for parents
        self.u = 0            # how many used. At forward, u+=1 per used. At backward, u-=1 per used. When u==0, grad becomes correct
        
    def __repr__(self):
        return "{}({})".format(__class__.__name__, self.func)
        
    def __add__(self, other):
        def f(x, y):
            return x + y
        def g(x, y):
            return (1.0, 1.0)
        return _define_binary(f, g, self, other)

    def __radd__(self, other):
        def f(x, y):
            return y + x
        def g(x, y):
            return (1.0, 1.0)
        return _define_binary(f, g, self, other)

    def __mul__(self, other):
        def f(x, y):
            return x * y
        def g(x, y):
            return (y, x)
        return _define_binary(f, g, self, other)

    def __rmul__(self, other):
        def f(x, y):
            return y * x
        def g(x, y):
            return (y, x)
        return _define_binary(f, g, self, other)
    
    def __sub__(self, other):
        def f(x, y):
            return x - y
        def g(x, y):
            return (1.0, -1.0)
        return _define_binary(f, g, self, other)

    def __rsub__(self, other):
        def f(x, y):
            return y - x
        def g(x, y):
            return (-1.0, 1.0)
        return _define_binary(f, g, self, other)

    def __pow__(self, other):
        def f(x, y):
            return x ** y
        def g(x, y):
            return (y * x ** (y-1), np.log(x) * x ** y)
        return _define_binary(f, g, self, other)
    
    def __rpow__(self, other):
        def f(x, y):
            return y ** x
        def g(x, y):
            return (np.log(y) * y ** x, x * y ** (x - 1))
        return _define_binary(f, g, self, other)

    def __truediv__(self, other):
        def f(x, y):
            return x / y
        def g(x, y):
            return (1.0 / y, - x / (y * y))
        return _define_binary(f, g, self, other)

    def __rtruediv__(self, other):
        def f(x, y):
            return y / x
        def g(x, y):
            return (- y / (x * x), 1.0 / x)
        return _define_binary(f, g, self, other)

    def __neg__(self):
        self.u += 1
        return BackNumber(-self.func, (-1.0,), (self,))

    def __lt__(self, other):
        return self.func.__lt__(other)

    def __gt__(self, other):
        return self.func.__gt__(other)

    def __le__(self, other):
        return self.func.__le__(other)

    def __ge__(self, other):
        return self.func.__ge__(other)
    
    def exp(self):
        return BackNumber(np.exp(self.func), (np.exp(self.func),), (self,))
    
    def log(self):
        return BackNumber(np.log(self.func), (1 / self.func,), (self,))

    def sqrt(self):
        return BackNumber(np.sqrt(self.func), (0.5 * self.func ** (-0.5),), (self,))
    
    def backward(self):
        self.grad = 1.0
        Q = deque([self])
        while Q:
            v = Q.popleft()
            for p, c in zip(v.p, v.c):
                p.grad += c * v.grad
                p.u -= 1
                if p.u == 0:
                    Q.append(p)
                
def gradient_numerical(f, inputs, dx):
    def delta(i, d):
        return [inp + d if i == j else inp for j, inp in enumerate(inputs)]
    return [(f(*delta(i, dx)) - f(*delta(i, -dx))) / (2 * dx) for i in range(len(inputs))]

def gradient_backward(f, inputs):
    v = [BackNumber(i) for i in inputs]
    z = f(*v)
    z.backward()
    return [vi.grad for vi in v]

def gradient_check(f, inputs, dx, tol):
    gn = gradient_numerical(f, inputs, dx)
    gb = gradient_backward(f, inputs)
    for i, (n, b) in enumerate(zip(gn, gb)):
        if abs((n - b) / b) > tol:
            return i
    return -1
