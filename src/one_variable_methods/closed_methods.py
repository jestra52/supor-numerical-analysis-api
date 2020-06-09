
from shared import util
from shared.function_manager import FunctionManager
import numpy as np
import pandas as pd

class ClosedMethods:
    def __init__(self, fx):
        self.__fx = FunctionManager(fx)

    def set_fx(self, fx):
        self.__fx = FunctionManager(fx)

    """
    Incremental search method
    """
    def incr_search(self, x0, delta, n):
        result = {
            'aproxValue': { 'x0': None, 'x1': None },
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'x': None,
                'fx': None
            }
        }
        fx0 = np.float64(self.__fx.eval_fx(x0))
        x0 = np.float64(x0)
        result['values']['fx'] = np.array([fx0])
        result['values']['x'] = np.array([x0])

        if delta <= 0 or n <= 0:
            result['error'] = True
            result['errorMessage']= 'Delta and iterations must be greater than 0'
        elif fx0 == 0:
            result['root'] = x0
            result['isExact'] = True
            result['resultMessage'] = f'x0={x0} is the root'
        else:
            x1 = x0 + delta
            fx1 = np.float64(self.__fx.eval_fx(x1))
            i = 0

            result['values']['x'] = np.append(result.get('values').get('x'), [np.float64(x1)])
            result['values']['fx'] = np.append(result.get('values').get('fx'), [fx1])

            # n-2 because there are already 2 rows assigned
            while fx0*fx1 > 0 and i < n-2:
                x0 = x1
                fx0 = fx1
                x1 = x0 + delta
                fx1 = np.float64(self.__fx.eval_fx(x1))
                i += 1

                result['values']['x'] = np.append(result.get('values').get('x'), [np.float64(x1)])
                result['values']['fx'] = np.append(result.get('values').get('fx'), [fx1])

            if fx1 == 0:
                result['root'] = x1
                result['resultMessage'] = f'x1={x1} is the root'
                result['isExact'] = True
            elif fx0*fx1 < 0:
                result['aproxValue'] = { 'x0': x0, 'x1': x1 }
                result['resultMessage'] = f'There is a root in [{x0}, {x1}]'
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True

        result['n'] = len(result.get('values').get('x'))
        return result

    """
    Bisection and false rule methods
    """
    def bisec_false_rule(self, xi, xs, tol, is_rel_error, is_false_rule, n):
        result = {
            'aproxValue': { 'x0': None, 'x1': None },
            'error': False,
            'errorMessage': None,
            'isExact': False,
            'n': 0,
            'resultMessage': None,
            'root': None,
            'solutionFailed': False,
            'values': {
                'xi': None,
                'xs': None,
                'xm': None,
                'fxm': None,
                'error': None,
            }
        }
        fxi = np.float64(self.__fx.eval_fx(xi))
        fxs = np.float64(self.__fx.eval_fx(xs))
        xm = np.float64(util.getClosedIntermediateValue(xi, xs, fxi, fxs, is_false_rule))
        fxm = np.float64(self.__fx.eval_fx(xm))
        result['values']['fxm'] = np.array([fxm])
        result['values']['xi'] = np.array([xi])
        result['values']['xs'] = np.array([xs])
        result['values']['xm'] = np.array([xm])
        result['values']['error'] = None

        if tol > 1:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be minor or equals to 1'
        elif n <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        elif fxi == 0:
            result['root'] = xm
            result['isExact'] = True
            result['resultMessage'] = f'xi={xi} is the root'
        elif fxi*fxs < 0:
            i = 0
            error = tol + 1

            # n-1 because there are already 1 row assigned
            while error > tol and fxm != 0 and i < n - 1:
                if fxi*fxm < 0:
                    xs = xm
                    fxs = fxm
                else:
                    xi = xm
                    fxi = fxm

                x_tmp = xm
                xm = np.float64(util.getClosedIntermediateValue(xi, xs, fxi, fxs, is_false_rule))
                fxm = np.float64(self.__fx.eval_fx(xm))
                error = np.float64(util.getError(xm, x_tmp, is_rel_error))
                i += 1

                result['values']['xi'] = np.append(result.get('values').get('xi'), [xi])
                result['values']['xs'] = np.append(result.get('values').get('xs'), [xs])
                result['values']['xm'] = np.append(result.get('values').get('xm'), [xm])
                result['values']['fxm'] = np.append(result.get('values').get('fxm'), [fxm])
                result['values']['error'] = np.append(result.get('values').get('error'), [error])

            if fxm == 0:
                result['root'] = xm
                result['resultMessage'] = f'xm={xm} is the root'
                result['isExact'] = True
            elif error < tol:
                result['aproxValue'] = { 'xi': xi, 'xs': xs }
                result['root'] = xm
                result['resultMessage'] = f'xm={xm} is an approximation to a root with a tolerance={tol}'
            else:
                result['resultMessage'] = f'Solution failed for n={n}'
                result['solutionFailed'] = True
        else:
            result['errorMessage'] = f'Interval is unsuitable'
            result['error'] = True

        result['n'] = len(result.get('values').get('xi'))
        return result
