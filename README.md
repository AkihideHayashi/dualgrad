
# The manual of dualgrad

Dualgrad is a simple gradient and part of hessian calculater for scalar functions.  

This library is "define by run".  
And it quickly creates calculation graph.  
So it is usefull when you want to calculate gradient and hessian for scalar functions such that the building of calculation graph is the rate determining step.  


```python
from dualgrad import BackNumber, DualNumber, define_function, define_functions, gradient_numerical
```

BackNumber is Class for backpropergation like Variable in chainer.  
DualNumber is Class for forwardpropergation using dual number.  
define_function(s) is a function that help generate function that is compatible for this library.


```python
import numpy as np
from numpy import sqrt, exp, log
import importlib
```

First, you should define your function for sympy.  
"define_function" uses sympy to differentiate your function analytically.


```python
def test_function(a, b, c):
    return (a + b) * c * a
```

To define your own function, you should use define_function.  
And we calculate numerical defferential only for tutorial.


```python
print(test_function(2.0, 3.0, 0.3))
print(gradient_numerical(test_function, (2.0, 3.0, 0.3), dx=1E-6))
```

    3.0
    [2.1000000003379427, 0.6000000001282757, 9.999999999621423]


In this case,


a = 2.0, b = 3.0, c = 0.3
test_function(a, b, c) = 3.0  
d(test_function)/da = 2.1  
d(test_function)/db = 0.6  

You can dump your function to a file if you want.


```python
with open("functions.py", 'w') as f:
    test_function_definition = define_function("test", test_function, ("a", "b"), ("c",))
    f.write(define_functions([test_function_definition]))
```


```python
import functions
from functions import test
importlib.reload(functions)
```




    <module 'functions' from '/Users/akihide/Documents/Development/dualgrad/Desktop/functions.py'>



Now, you imported "test" that is compatible for this library.
You can calculate differential of "test" quickly using BackNumber.


```python
x = BackNumber(2.0)
y = BackNumber(3.0)
w = 0.3
z = test(x, y, w)
z
```




    BackNumber(3.0)




```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

    2.1
    0.6


If you need part of hessian, you should use DualNumber to initialize BackNumber.


```python
x = BackNumber(DualNumber(2.0, np.array([1.0, 0.0])))
y = BackNumber(DualNumber(3.0, np.array([0.0, 1.0])))
w = 0.3
z = test(x, y, w)
z
```




    BackNumber(DualNumber(3.0, [2.1 0.6]))




```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

    DualNumber(2.1, [0.6 0.3])
    DualNumber(0.6, [0.3 0. ])


now,  
d^2z/dx^2 = 0.6  
d^2z/dxdy = 0.3  
d^2z/dy^2 = 0.0

While simple operators are implemented. You can skip function_difiner


```python
x = BackNumber(DualNumber(2.0, np.array([1.0, 0.0])))
y = BackNumber(DualNumber(3.0, np.array([0.0, 1.0])))
z = x * x + 0.5 * y * y + x * y
z
```




    BackNumber(DualNumber(14.5, [7. 5.]))




```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

    DualNumber(7.0, [2. 1.])
    DualNumber(5.0, [1. 1.])


Known problems: We cant calculate hessian when the part of hessian is all 0
