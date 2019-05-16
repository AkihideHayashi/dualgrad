
# The manual of dualgrad

Dualgrad is a simple gradient and part of hessian calculater for scalar functions.  

This library is "define by run".  
And it quickly creates calculation graph.  
So it is usefull when you want to calculate gradient and hessian for scalar functions such that the building of calculation graph is the rate determining step.  


```python
from dualgrad import BackNumber, DualNumber, gradient_numerical, cythonize, pythonize, define_cython, define_python
```

BackNumber is Class for backpropergation like Variable in chainer.  
DualNumber is Class for forwardpropergation using dual number.  


```python
import numpy as np
from numpy import sqrt, exp, log
import sympy
from sympy import symbols
import importlib
```

You can calculate gradient using backpropergation.


```python
x = BackNumber(2.0)
y = BackNumber(3.0)
z = x * x + 0.5 * y * y + x * y
print(z.func)
z.backward()
print(x.grad, y.grad)
```

    14.5
    7.0 5.0


z = 14.5, dz/dx = 7.0, dz/dy = 5.0

You can also calculate second derivative using combination of backpropergation and dual number


```python
x = BackNumber(DualNumber(2.0, 1.0))
y = BackNumber(DualNumber(3.0, 0.0))
z = x * x + 0.5 * y * y + x * y
z.backward()
print(x.grad.dual, y.grad.dual)
```

    2.0 1.0


d^2z/dxdx = 2.0, d^2z/dxdy = 1.0

However, when you construct huge function, creation of function node be able to peformance determining step.  
So dualgrad supports method for skip the process.


```python
def test_function(a, b, c):
    return (a + b) * c * a * a + b / c / a + b + c * exp(a) * log(a)

print(test_function(2.0, 3.0, 0.3))
print(gradient_numerical(test_function, (2.0, 3.0, 0.3), dx=1E-6))
```

    15.536511020591915
    [7.344869436209933, 3.86666666685187, 8.45503673563286]


test function is a function that when
a = 2.0, b = 3.0, c = 0.3,

test_function(a, b, c) = 15.536511020591915  
d(test_function)/da ~ 7.344869436209933  
d(test_function)/db ~ 3.86666666685187  
d^2(test_function)/dada ~ 11.09904864285131  
d^2(test_function)/dbda ~ 0.3666666666666666  


```python
a = BackNumber(DualNumber(2.0, 1.0))
b = BackNumber(DualNumber(3.0, 0.0))
c = 0.3
z = test_function(a, b, c)
z.backward()
print(z.func.real)
print(a.grad.real, b.grad.real)
print(a.grad.dual, b.grad.dual)
```

    15.536511020591915
    7.344869435431512 3.866666666666667
    11.09904864285131 0.3666666666666666


However, it is very slow.


```python
%%timeit
a = BackNumber(DualNumber(2.0, 1.0))
b = BackNumber(DualNumber(3.0, 0.0))
c = 0.3
z = test_function(a, b, c)
z.backward()
```

    113 µs ± 6.46 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


It is possible to simplify and dump funciton.


```python
def test_function(a, b, c):
    return (a + b) * c * a * a + b / c / a + b + c * sympy.exp(a) * sympy.log(a)
```


```python
with open("functions.py", 'w') as f:
    test_function_definition = pythonize("test", test_function, symbols("a, b"), symbols("c,"))
    f.write(define_python([test_function_definition]))
```


```python
import functions
from functions import test
importlib.reload(functions)
```






Now, you imported "test" that is compatible for this library.
You can calculate differential of "test" quickly using BackNumber.


```python
%%timeit
a = BackNumber(DualNumber(2.0, 1.0))
b = BackNumber(DualNumber(3.0, 0.0))
c = 0.3
z = test(a, b, c)
z.backward()
```

    72.9 µs ± 132 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


it is a little faster than doing nothing. However it is not so fast

It is also possible to cythonize function.


```python
with open("cfunctions.pyx", 'w') as f:
    test_function_definition = cythonize("test", test_function, symbols("a, b"), symbols("c,"))
    f.write(define_cython([test_function_definition]))
```


```python
import pyximport; pyximport.install()
from cfunctions import test
```


```python
%%timeit
a = BackNumber(DualNumber(2.0, 1.0))
b = BackNumber(DualNumber(3.0, 0.0))
c = 0.3
z = test(a, b, c)
z.backward()
```

    10.4 µs ± 198 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


cythonize only support DualNumber now. So you can't skip calculating partial hessian.


```python

```
