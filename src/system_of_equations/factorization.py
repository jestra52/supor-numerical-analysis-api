import copy as cp
import math as mt
import numpy as np
import threading

class Factorization:
    def cholesky(self, A, b):
        result = {
            'aMatrix': None,
            'bMatrix': None,
            'lMatrix': None,
            'uMatrix': None,
            'xMatrix': None,
            'iterations': None,
            'hasInfiniteSolutions': False,
            'resultMessage': None,
            'solutionFailed': False,
            'error': False,
            'errorMessage': None
        }
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        phases = list()

        def diagonal_operation_async(k):
            incr = 0
            for p in range(0, k):
                incr += L[k][p] * U[p][k]
            L[k][k] = mt.sqrt(A[k][k] - incr)
            U[k][k] = L[k][k]

        def row_operation_async(k, i):
            incr = 0
            for r in range(0, k):
                incr += L[i][r] * U[r][k]
            L[i][k] = (A[i][k] - incr) / L[k][k]

        def column_operation_async(k, j):
            incr = 0
            for s in range(0, k):
                incr += L[k][s] * U[s][j]
            U[k][j] = (A[k][j] - incr) / L[k][k]

        for k in range(0, n):
            thread = threading.Thread(target=diagonal_operation_async, args=([k]))
            thread.start()
            thread.join()

            if L[k][k] == 0:
                raise ZeroDivisionError
            
            threads = list()
            for i in range(k+1, n):
                thread = threading.Thread(target=row_operation_async, args=(k, i))
                threads.append(thread)
                thread.start()
            for thread in threads: thread.join()
            
            threads.clear()
            for j in range(k+1, n):
                thread = threading.Thread(target=column_operation_async, args=(k, j))
                threads.append(thread)
                thread.start()
            for thread in threads: thread.join()

            if k < n - 1:
                iteration = {
                    'lMatrix': list(map(lambda l: list(l), cp.deepcopy(L))),
                    'uMatrix': list(map(lambda u: list(u), cp.deepcopy(U))),
                }
                phases.append(cp.deepcopy(iteration))

        if not result['error']:
            result['aMatrix'] = A
            result['bMatrix'] = b
            result['lMatrix'] = L
            result['uMatrix'] = U
            result['xMatrix'] = self.solve_x(L, U, b)
            result['iterations'] = phases

        return result

    def doolittle(self, A, b):
        result = {
            'aMatrix': None,
            'bMatrix': None,
            'lMatrix': None,
            'uMatrix': None,
            'xMatrix': None,
            'iterations': None,
            'hasInfiniteSolutions': False,
            'resultMessage': None,
            'solutionFailed': False,
            'error': False,
            'errorMessage': None
        }
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        phases = list()

        def column_operation_async(k, j):
            incr = 0
            for p in range(k):
                incr += L[k][p] * U[p][j]
            U[k][j] = (A[k][j] - incr)

        def row_operation_async(k, i):
            incr = 0
            for r in range(k):
                incr += L[i][r] * U[r][k]
            L[i][k] = (A[i][k] - incr) / U[k][k]

        for k in range(0,n):
            threads = list()
            for j in range(k, n):
                thread = threading.Thread(target=column_operation_async, args=(k, j))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()

            if U[k][k] == 0:
                raise ZeroDivisionError
            
            threads.clear()
            for i in range(k, n):
                thread = threading.Thread(target=row_operation_async, args=(k, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if k < n - 1:
                iteration = {
                    'lMatrix': list(map(lambda l: list(l), cp.deepcopy(L))),
                    'uMatrix': list(map(lambda u: list(u), cp.deepcopy(U))),
                }
                phases.append(cp.deepcopy(iteration))

        if not result['error']:
            result['aMatrix'] = A
            result['bMatrix'] = b
            result['lMatrix'] = L
            result['uMatrix'] = U
            result['xMatrix'] = self.solve_x(L, U, b)
            result['iterations'] = phases

        return result

    def crout(self, A, b):
        result = {
            'aMatrix': None,
            'bMatrix': None,
            'lMatrix': None,
            'uMatrix': None,
            'xMatrix': None,
            'iterations': None,
            'hasInfiniteSolutions': False,
            'resultMessage': None,
            'solutionFailed': False,
            'error': False,
            'errorMessage': None
        }
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        phases = list()

        def row_operation_async(k, i):
            incr = 0
            for p in range(0,k):
                incr += L[i][p] * U[p][k]
            L[i][k] = A[i][k] - incr

        def column_operation_async(k, j):
            incr = 0
            for p in range(0,k):
                incr += L[k][p] * U[p][j]
            U[k][j] = (A[k][j] - incr) / L[k][k]

        for k in range(0, n):
            threads = list()
            for i in range(k, n):
                thread = threading.Thread(target=row_operation_async, args=(k, i))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()

            if L[k][k] == 0:
                raise ZeroDivisionError
            
            threads.clear()
            for j in range(k, n):
                thread = threading.Thread(target=column_operation_async, args=(k, j))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if k < n - 1:
                iteration = {
                    'lMatrix': list(map(lambda l: list(l), cp.deepcopy(L))),
                    'uMatrix': list(map(lambda u: list(u), cp.deepcopy(U))),
                }
                phases.append(cp.deepcopy(iteration))

        if not result['error']:
            result['aMatrix'] = A
            result['bMatrix'] = b
            result['lMatrix'] = L
            result['uMatrix'] = U
            result['xMatrix'] = self.solve_x(L, U, b)
            result['iterations'] = phases

        return result

    def solve_z(self, L, b):
        n = len(b)
        Z = []

        for i in range(n):
            Z.append(0)

        for i in range(0, n):
            incr = 0

            for p in range(0, i):
                incr += L[i][p] * Z[p]

            if L[i][i] == 0:
                raise ZeroDivisionError

            Z[i] = (b[i] - incr) / L[i][i]

        return Z

    def solve_x(self, L, U, b):
        n = len(b)
        Z = self.solve_z(L, b)
        X = []

        for i in range(n):
            X.append(0)

        i = n - 1

        while i >= 0:
            incr = 0

            for p in range(i+1, n):
                incr += U[i][p] * X[p]

            if U[i][i] == 0:
                raise ZeroDivisionError

            X[i] = (Z[i] - incr) / U[i][i]
            i -= 1

        return X

    def get_invertible_matrix(self, L, U):
        n = len(L)
        invertible_a = []

        for i in range(0, n):
            b = []

            for j in range(0, n):
                if j == i: b.append(1)
                else: b.append(0)

            invertible_a.append(self.solve_x(L, U, b))

        return invertible_a
