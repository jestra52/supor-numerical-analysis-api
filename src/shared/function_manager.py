from sympy import Symbol, diff, expand
from sympy.parsing.sympy_parser import parse_expr

class FunctionManager:
    def __init__(self, fx):
        self.__x = Symbol('x')
        self.__fx = parse_expr(fx)
        self.__dfdx = diff(fx, self.__x)

    def eval_fx(self, x):
        return self.__fx.subs(self.__x, x)

    def eval_dfdx(self, x):
        return self.__dfdx.subs(self.__x, x)

    def diff_dfdx(self):
        self.__dfdx = diff(self.__dfdx, self.__x)
        return self.__dfdx

    def get_fx(self):
        return self.__fx

    def get_dfdx(self):
        return self.__dfdx
