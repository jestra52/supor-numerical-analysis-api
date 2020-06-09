from shared import util
import copy
import math as mt
import numpy as np
import threading

class IterativeMethods():
    def gauss_seidel(self, A, b, iterations, tol, l, xi, is_rel_error):
        x = copy.deepcopy(xi)
        result = {
            'error': False,
            'errorMessage': None,
            'n': 0,
            'resultMessage': None,
            'solutionFailed': False,
            'values': self.build_values(x)
        }
        if tol < 0:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be greater or equals than 0'
        elif iterations <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        else:
            m = len(x)
            xold = []

            for i in range(m):
                xold.append(x[i])
                x[i] = self.solve_new_x(i,x,l,A,b)
                result['values'][f'x{i}'] = np.append(result.get('values').get(f'x{i}'), [x[i]])

            error = np.float(util.getMatrixError(is_rel_error, xold, x))
            result['values']['error'] = None
            n=1

            while error > tol and n < iterations:
                xold = []

                for i in range(m):
                    xold.append(x[i])
                    x[i] = self.solve_new_x(i,x,l,A,b)
                    result['values'][f'x{i}'] = np.append(result.get('values').get(f'x{i}'), [x[i]])

                error = np.float(util.getMatrixError(is_rel_error, xold, x))
                result['values']['error'] = np.append(result.get('values').get('error'), [error])
                n+=1

            if error < tol:
                result['resultMessage'] = f'The solution was successful with a tolerance={tol} and {n} iterations'
            else:
                result['resultMessage'] = f'Solution failed for n={n} iterations'
                result['solutionFailed'] = True
        result['n'] = len(result.get('values').get('x0'))
        return result

    def jacobi(self, A, b, iterations, tol, l, x, is_rel_error):
        result = {
            'error': False,
            'errorMessage': None,
            'n': 0,
            'resultMessage': None,
            'solutionFailed': False,
            'values': self.build_values(x)
        }
        if tol < 0:
            result['error'] = True
            result['errorMessage']= 'Tolerance must be greater or equals than 0'
        elif iterations <= 0:
            result['error'] = True
            result['errorMessage']= 'Iterations must be greater than 0'
        else:
            m = len(x)
            n = 0
            error = tol + 1
            while error>tol and n < iterations:
                xnew = np.zeros(m,dtype=np.float64)
                
                #PARALLELISM
                threads = list()
                for var_index in range(m):
                    thread = threading.Thread( target=self.solve_new_x_async, args=(A, b, var_index, x, xnew, l, m))
                    threads.append(thread)
                    thread.start()

                i = 0
                for thread in threads:
                    thread.join()
                    result['values'][f'x{i}'] = np.append(result.get('values').get(f'x{i}'), [xnew[i]])
                    i+=1

                error = np.float(util.getMatrixError(is_rel_error,x,xnew))
                result['values']['error'] = np.append(result.get('values').get('error'), [error])
                x = xnew
                n+=1

            if error < tol:
                result['resultMessage'] = f'The solution was successful with a tolerance={tol} and {n} iterations'
            else:
                result['resultMessage'] = f'Solution failed for n={n} iterations'
                result['solutionFailed'] = True
        result['n'] = len(result.get('values').get('x0'))
        return result

    def solve_new_x_async(self, A, b, i, xi, xn, lamb, n):
        j = 0
        den = 1
        sum = b[i]

        while den != 0 and j < n:
            if(j != i):
                sum -= A[i][j] * xi[j]
            else:
                den = A[i][j]
            j += 1
        if den != 0:
            xn[i] = (lamb*(sum/den) + (1-lamb)*xi[i])
        else:
            raise ZeroDivisionError

    def solve_new_x(self, i, arrX, l, A, b):
        n = len(A)
        den = 1
        incr = b[i]
        j = 0

        while j < n and den != 0:
            if i == j:
                den = A[i][j]
            else:
                incr += (-1) * A[i][j] * arrX[j]
            j+=1

        if den == 0:
            raise ZeroDivisionError
        else:
            xn  = incr / den
            xn = l * xn + (1 - l) * arrX[i]
            return xn

    def build_values(self, x):
        values = {}

        for i in range(len(x)):
            values[f'x{i}'] = np.array([x[i]])

        values["error"] = None

        return values

    def best_lambda_j(self, A, b, x, is_rel_error):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'lambda': None
        }
        itv = [0,1]
        prop = 0.5
        resultc = self.jacobi(A, b, 100, 0.000001, prop, x, is_rel_error)
        if resultc['n'] <= 2:
            result['error'] = True
            result['errorMessage']= 'to calculate lambda the iterations must be greater than 2'
        else:

            while itv[0] != itv[1]:
                results = self.jacobi(A, b, 100, 0.000001, prop+0.01, x, is_rel_error)
                resulti = self.jacobi(A, b, 100, 0.000001, prop-0.01, x, is_rel_error)
                ns = results['n']
                ni = resulti['n']
                es = results['values']['error'][-1]
                ei = resulti['values']['error'][-1]
                if ns < ni:
                    itv = [prop+0.01,itv[1]]
                    prop = round((itv[0]+itv[1])/2,2)
                elif ni < ns:
                    itv = [itv[0],prop-0.01]
                    prop = round((itv[0]+itv[1])/2,2)
                elif es < ei:
                    result['resultMessage'] = f'We recommend to use lambda = {prop} in jacobi'
                    result['lambda'] = prop
                    return result
                else:
                    result['resultMessage'] = f'We recommend to use lambda = {prop} in jacobi'
                    result['lambda'] = prop
                    return result

        result['resultMessage'] = f'We recommend to use lambda = {prop} in jacobi'
        result['lambda'] = prop
        return result

    def best_lambda_g(self, A, b, x, is_rel_error):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'lambda': None
        }
        itv = [0,1]
        prop = 0.5
        resultc = self.gauss_seidel(A, b, 100, 0.000001, prop, x, is_rel_error)
        if resultc['n'] <= 2:
            result['error'] = True
            result['errorMessage']= 'to calculate lambda the iterations must be greater than 2'
        else:

            while itv[0] != itv[1]:
                results = self.gauss_seidel(A, b, 100, 0.000001, prop+0.01, x, is_rel_error)
                resulti = self.gauss_seidel(A, b, 100, 0.000001, prop-0.01, x, is_rel_error)
                ns = results['n']
                ni = resulti['n']
                es = results['values']['error'][-1]
                ei = resulti['values']['error'][-1]

                if ns < ni:
                    itv = [prop+0.01,itv[1]]
                    prop = round((itv[0]+itv[1])/2,2)
                elif ni < ns:
                    itv = [itv[0],prop-0.01]
                    prop = round((itv[0]+itv[1])/2,2)
                elif es < ei:
                    result['resultMessage'] = f'We recommend to use lambda = {prop} in gauss seidel'
                    result['lambda'] = prop
                    return result
                else:
                    result['resultMessage'] = f'We recommend to use lambda = {prop} in gauss seidel'
                    result['lambda'] = prop
                    return result

        result['resultMessage'] = f'We recommend to use lambda = {prop} in gauss seidel'
        result['lambda'] = prop
        return result
