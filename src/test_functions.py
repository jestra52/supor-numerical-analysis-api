import numpy as np

def test_fx_0(x):
    return x*np.exp(-(x**2)+1) - 6*x*np.cos(4*x-3) + 8*x - 10

def test_fx_1(x):
    return x*np.exp(-x+2) - 5*x*np.sin(x) - 7

def test_fx_2(x):
    return np.exp(3*x-12) + x*np.cos(3*x) - x**2 +4

def test_fx_3(x):
    return x*np.exp(x) - x**2 - 5*x - 3

def test_fx_4(x):
    return np.exp(-x) - (x**2)*np.cos(2*x-4) + 6*x + 3

def test_fx_5(x):
    return np.exp(x) - 5*x + 2

def test_fx_6(x):
    return np.exp(-(x**2)+1) - x*np.sin(2*x+3) - 4*x +4

def test_gx_0(x):
    return np.log((x**2 + 5*x + 3)/x)

def test_gx_1(x):
    return (x*np.exp(x) - x**2 - 3)/5

def test_gx_2(x):
    return -np.sqrt(x*np.exp(x) - 5*x - 3)

def test_dfdx_0_fx_4(x):
    return -np.exp(-x) + 2*x*np.sin(2*x-4) - 2*x*np.cos(2*x-4) + 6

def test_parse_fx_0():
    return 'x*exp(-(x**2) + 1) - 6*x*cos(4*x-3) + 8 * x - 10'

def test_parse_fx_1():
    return 'cos(x) - x'

def test_parse_fx_2():
    return 'x*exp(3**x-12) - x*cos(3*x) - x**2 + 4'

def test_parse_fx_3():
    return 'x*exp(-x+2) - 5*x*sin(x) - 7'

def test_parse_gx_0():
    return '(x**2 + 5*x +3)/x'

def test_parse_gx_1():
    return 'cos(x)'
