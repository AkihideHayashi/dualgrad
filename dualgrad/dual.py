from numpy import exp, log, sqrt

def _define_binary(f, x, y):
    if isinstance(y, DualNumber):
        return DualNumber(*f(x.real, x.dual, y.real, y.dual))
    else:
        return DualNumber(*f(x.real, x.dual, y.real, 0))


class DualNumber(object):
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
        
    def __add__(self, other):
        def f(x, dx, y, dy):
            return x + y, dx + dy
        return _define_binary(f, self, other)
    
    __radd__ = __add__
    
    def __sub__(self, other):
        def f(x, dx, y, dy):
            return x - y, dx - dy
        return _define_binary(f, self, other)
    
    __rsub__ = __sub__
    
    def __mul__(self, other):
        def f(x, dx, y, dy):
            return x * y, x * dy + y * dx
        return _define_binary(f, self, other)
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        def f(x, dx, y, dy):
            return x / y, (dx * y - x * dy) / (y * y)
        return _define_binary(f, self, other)
        
    def __rtruediv__(self, other):
        def f(y, dy, x, dx):
            return x / y, (dx * y - x * dy) / (y * y)
        return _define_binary(f, self, other)
    
    def __pow__(self, other):
        def f(x, dx, y, dy):
            return x ** y, x ** (y - 1) * (x * log(x) * dy + y * dx)
        return _define_binary(f, self, other)
    
    def __rpow__(self, other):
        def f(y, dy, x, dx):
            return x ** y, x ** (y - 1) * (x * log(x) * dy + y * dx)
        return _define_binary(f, self, other)

    def __neg__(self):
        return __class__(-self.real, - self.dual)
        
    def __lt__(self, other):
        return self.real.__lt__(other)

    def __gt__(self, other):
        return self.real.__gt__(other)

    def __le__(self, other):
        return self.real.__le__(other)

    def __ge__(self, other):
        return self.real.__ge__(other)

    def exp(self):
        return __class__(exp(self.real), exp(self.real) * self.dual)
    
    def log(self):
        return __class__(log(self.real), self.dual / self.real)
        
    def sqrt(self):
        return __class__(sqrt(self.real), self.dual / (2 * sqrt(self.real)))
        
    def __repr__(self):
        return "{}({}, {})".format(__class__.__name__, self.real, self.dual)