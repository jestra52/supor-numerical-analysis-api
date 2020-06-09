from shared import util
from shared.function_manager import FunctionManager
from sympy import Symbol
import numpy as np
import pandas as pd

class OpenMethods:
    def __init__(self, fx, gx):
        self.__fx = FunctionManager(fx)
        self.__gx = FunctionManager(gx) if gx is not None else gx

    def set_fx(self, fx):
        self.__fx = FunctionManager(fx)

    def set_gx(self, gx):
        self.__gx = FunctionManager(gx)

    """
    Fixed-point iteration method
    """
    def fixed_point_iteration(self, xa, tol, is_rel_error, n):
        result = {
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'xn': None,
                'fxn': None,
                'error': None
            }
        }

        xa = np.float64(xa)
        fxn = np.float64(self.__fx.eval_fx(xa))
        result['values']['fxn'] = np.array([fxn])
        result['values']['xn'] = np.array([xa])
        result['values']['error'] = None

        if tol > 1:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be minor or equals to 1'
        elif n <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        elif fxn == 0:
            result['root'] = xa
            result['isExact'] = True
            result['resultMessage'] = f'x0={xa} is the root'
        else:
            i = 1
            error = tol + 1

            while fxn != 0 and error > tol and i < n:
                xn = np.float64(self.__gx.eval_fx(xa))
                fxn = np.float64(self.__fx.eval_fx(xn))
                error = np.float64(util.getError(xn, xa, is_rel_error))
                xa = xn
                i += 1

                result['values']['xn'] = np.append(result.get('values').get('xn'), [xn])
                result['values']['fxn'] = np.append(result.get('values').get('fxn'), [fxn])
                result['values']['error'] = np.append(result.get('values').get('error'), [error])

            if fxn == 0:
                result['root'] = xa
                result['resultMessage'] = f'xn={xa} is the root'
                result['isExact'] = True
            elif error < tol:
                result['root'] = xa
                result['resultMessage'] = f'xn={xa} is an approximation to a root with a tolerance={tol}'
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True

        result['n'] = len(result.get('values').get('xn'))
        return result

    """
    Newton's method
    """
    def newton(self, x0, tol, is_rel_error, n):
        result = {
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'xn': None,
                'fxn': None,
                'dfxn': None,
                'error': None
            }
        }

        x0 = np.float64(x0)
        fxn = np.float64(self.__fx.eval_fx(x0))
        dfxndx = self.__fx.eval_dfdx(x0)
        result['values']['fxn'] = np.array([fxn])
        result['values']['dfxn'] = np.array([fxn])
        result['values']['xn'] = np.array([x0])
        result['values']['error'] = None

        if tol > 1:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be minor or equals to 1'
        elif n <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        elif fxn == 0:
            result['root'] = x0
            result['isExact'] = True
            result['resultMessage'] = f'x0={x0} is the root'
        else:
            i = 1
            error = tol + 1

            while fxn != 0 and dfxndx != 0 and error > tol and i < n:
                xn = np.float64(x0 - fxn/dfxndx)
                fxn = np.float64(self.__fx.eval_fx(xn))
                dfxndx = np.float64(self.__fx.eval_dfdx(xn))
                error = np.float64(util.getError(xn, x0, is_rel_error))
                x0 = xn
                i += 1

                result['values']['xn'] = np.append(result.get('values').get('xn'), [x0])
                result['values']['fxn'] = np.append(result.get('values').get('fxn'), [fxn])
                result['values']['dfxn'] = np.append(result.get('values').get('dfxn'), [dfxndx])
                result['values']['error'] = np.append(result.get('values').get('error'), [error])

            if fxn == 0:
                result['root'] = xn
                result['resultMessage'] = f'xn={xn} is the root'
                result['isExact'] = True
            elif error < tol:
                result['root'] = xn
                result['resultMessage'] = f'xn={xn} is an approximation to a root with a tolerance={tol}'
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True

        result['n'] = len(result.get('values').get('xn'))
        return result

    """
    Secant method
    """
    def secant(self, x0, x1, tol, is_rel_error, n):
        result = {
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'xn': None,
                'fxn': None,
                'den': None,
                'error': None
            }
        }

        x0 = np.float64(x0)
        x1 = np.float64(x1)
        fx0 = np.float64(self.__fx.eval_fx(x0))
        fx1 = np.float64(self.__fx.eval_fx(x1))
        result['values']['fxn'] = np.array([fx0])
        result['values']['xn'] = np.array([x0])
        result['values']['error'] = None

        if tol > 1:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be minor or equals to 1'
        elif n <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        elif fx0 == 0:
            result['root'] = x0
            result['isExact'] = True
            result['resultMessage'] = f'x0={x0} is the root'
        elif fx1 == 0:
            result['root'] = x1
            result['isExact'] = True
            result['resultMessage'] = f'x1={x1} is the root'
        else:
            i = 0
            error = tol + 1
            den = np.float64(fx1 - fx0)
            result['values']['xn'] = np.append(result.get('values').get('xn'), [x1])
            result['values']['fxn'] = np.append(result.get('values').get('fxn'), [fx1])
            result['values']['den'] = np.append(result.get('values').get('den'), [None, den])
            result['values']['error'] = np.append(result.get('values').get('error'), [None])

            # n-1 because there are already 1 row assigned
            while fx1 != 0 and den != 0 and error > tol and i < n - 2:
                x2 = np.float64(x1 - fx1*(x1-x0)/den)
                error = np.float64(util.getError(x2, x1, is_rel_error))
                x0 = x1
                fx0 = fx1
                x1 = x2
                fx1 = np.float64(self.__fx.eval_fx(x1))
                den = np.float64(fx1 - fx0)
                i += 1

                result['values']['xn'] = np.append(result.get('values').get('xn'), [x1])
                result['values']['fxn'] = np.append(result.get('values').get('fxn'), [fx1])
                result['values']['den'] = np.append(result.get('values').get('den'), [den])
                result['values']['error'] = np.append(result.get('values').get('error'), [error])

            if fx1 == 0:
                result['root'] = x1
                result['resultMessage'] = f'x1={x1} is the root'
                result['isExact'] = True
            elif error < tol:
                result['root'] = x1
                result['resultMessage'] = f'x1={x1} is an approximation to a root with a tolerance={tol}'
            elif den == 0:
                result['resultMessage'] = 'There is a possible multiple root'
                result['solutionFailed'] = True
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True

        result['n'] = len(result.get('values').get('xn'))
        return result

    """
    Multiple roots method
    """
    def multiple_roots(self, x0, tol, is_rel_error, n):
        result = {
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'd2fxn': None,
                'dfxn': None,
                'error': None,
                'fxn': None,
                'xn': None
            }
        }

        x = Symbol('x')
        x0 = np.float64(x0)
        fxn = np.float64(self.__fx.eval_fx(x0))
        dfdx0 = self.__fx.get_dfdx()
        dfdxfx0 = np.float64(dfdx0.subs(x, x0))
        dfdx1 = self.__fx.diff_dfdx()
        dfdxfx1 = np.float64(dfdx1.subs(x, x0))

        result['values']['xn'] = np.array([x0])
        result['values']['fxn'] = np.array([fxn])
        result['values']['dfxn'] = np.array([dfdxfx0])
        result['values']['d2fxn'] = np.array([dfdxfx1])
        result['values']['error'] = None

        if tol > 1:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be minor or equals to 1'
        elif n <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        elif fxn == 0:
            result['root'] = x0
            result['isExact'] = True
            result['resultMessage'] = f'x0={x0} is the root'
        else:
            i = 0
            error = tol + 1
            den = np.float64(dfdxfx0**2 - (fxn*dfdxfx1))

            while fxn != 0 and den != 0 and error > tol and i < n:
                xn = np.float64(x0 - fxn*dfdxfx0/den)
                error = np.float64(util.getError(xn, x0, is_rel_error))
                x0 = xn
                fxn = np.float64(self.__fx.eval_fx(xn))
                dfdxfx0 = np.float64(dfdx0.subs(x, xn))
                dfdxfx1 = np.float64(dfdx1.subs(x, xn))

                result['values']['xn'] = np.append(result.get('values').get('xn'), [x0])
                result['values']['fxn'] = np.append(result.get('values').get('fxn'), [fxn])
                result['values']['dfxn'] = np.append(result.get('values').get('dfxn'), [dfdxfx0])
                result['values']['d2fxn'] = np.append(result.get('values').get('d2fxn'), [dfdxfx1])
                result['values']['error'] = np.append(result.get('values').get('error'), [error])

                den = np.float64(dfdxfx0**2 - (fxn*dfdxfx1))
                i += 1

            if fxn == 0:
                result['root'] = x0
                result['resultMessage'] = f'x0={x0} is the root'
                result['isExact'] = True
            elif error < tol:
                result['root'] = xn
                result['resultMessage'] = f'xn={xn} is an approximation to a root with a tolerance={tol}'
            elif den == 0:
                result['resultMessage'] = 'There is a possible multiple root'
                result['solutionFailed'] = True
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True

        result['n'] = len(result.get('values').get('xn'))
        return result
