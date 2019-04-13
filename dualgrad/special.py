from .back import BackNumber
from .dual import DualNumber
import numpy as np
from numpy import exp, sqrt, pi
from scipy.special import erf as sp_erf  # pylint: disable=no-name-in-module

def erf(x):
    if isinstance(x, BackNumber):
        for node in (x,):
            node.u += 1
        return BackNumber(sp_erf(x.x), (x,), (2*exp(-x.x**2)/sqrt(pi),))
    elif isinstance(x, DualNumber):
        return DualNumber(sp_erf(x.real), 2*exp(-x.real**2)/sqrt(pi) * x.dual)
    else:
        return sp_erf(x)