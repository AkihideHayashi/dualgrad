
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
print(test_definer(2.0, 3.0, 0.3))
print(gradient_numerical(test_definer, (2.0, 3.0, 0.3), dx=1E-6))
```

In this case,

```math
$$a = 2.0, b = 3.0, c = 0.3$$

$$\mathrm{test\_function}(a, b, c) = 3.0$$

$$\frac{\mathrm{d}(\mathrm{test\_function})}{\mathrm{d}a} = 2.1$$  
$$\frac{\mathrm{d}(\mathrm{test\_function})}{\mathrm{d}b} = 0.6$$
```

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

Now, you imported "test" that is compatible for this library.
You can calculate differential of "test" quickly using BackNumber.


```python
x = BackNumber(2.0)
y = BackNumber(3.0)
w = 0.3
z = test(x, y, w)
z
```


```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

If you need part of hessian, you should use DualNumber to initialize BackNumber.


```python
x = BackNumber(DualNumber(2.0, np.array([1.0, 0.0])))
y = BackNumber(DualNumber(3.0, np.array([0.0, 1.0])))
w = 0.3
z = test(x, y, w)
z
```


```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

now,  

```
$\frac{d^2z}{dx^2} = 0.6$, $\frac{d^2z}{dxdy} = 0.3$, $\frac{d^2z}{dy^2} = 0.0$
```

While simple operators are implemented. You can skip function_difiner


```python
x = BackNumber(DualNumber(2.0, np.array([1.0, 0.0])))
y = BackNumber(DualNumber(3.0, np.array([0.0, 1.0])))
z = x * x + 0.5 * y * y + x * y
z
```


```python
z.backward()
```


```python
print(x.grad)
print(y.grad)
```

Known problems: We cant calculate hessian when the part of hessian is all 0


```python

```
