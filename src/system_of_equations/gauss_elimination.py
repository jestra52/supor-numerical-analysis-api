from shared import util
import copy as cp
import numpy as np

np.seterr(divide='raise')

class GaussElimination:
    def execute(self, A, b, method_type = None):
        result = {
            'aMatrix': None,
            'bMatrix': None,
            'xMatrix': None,
            'iterations': None,
            'sortingIterations': None,
            'hasInfiniteSolutions': False,
            'resultMessage': None,
            "solutionFailed": False,
            'error': False,
            'errorMessage': None,
        }
        ab = []
        iterations = list()
        sorting_iterations = list()
        n = len(A)
        marks = []

        if method_type == util.method_type.get('GAUSS_COMPLETE'):
            for i in range(n+1):
                marks.append(i+1)

        for k in range(1, n):
            if method_type == util.method_type.get('GAUSS_PARTIAL'):
                step = { 'aMatrix': cp.deepcopy(A), 'bMatrix': cp.deepcopy(b) }
                sorting_iterations.append(cp.deepcopy(step))
                ab = util.augmented_matrix(cp.deepcopy(A), cp.deepcopy(b))
                ab = self.partial_pivoting(cp.deepcopy(ab), n, k-1)

                if len(ab) == 0:
                    result['error'] = True
                    result['hasInfiniteSolutions'] = True
                    result['solutionFailed'] = True
                    result['errorMessage'] = 'The system has infinite solutions'
                    break
                else:
                    A, b = util.split_system(cp.deepcopy(ab))
                    step = { 'aMatrix': cp.deepcopy(A), 'bMatrix': cp.deepcopy(b) }
                    sorting_iterations.append(cp.deepcopy(step))

            if method_type == util.method_type.get('GAUSS_COMPLETE'):
                step = { 'aMatrix': cp.deepcopy(A), 'bMatrix': cp.deepcopy(b) }
                sorting_iterations.append(cp.deepcopy(step))
                ab = util.augmented_matrix(cp.deepcopy(A), cp.deepcopy(b))
                ab = self.complete_pivoting(cp.deepcopy(ab), n, k-1, cp.deepcopy(marks))

                if len(ab) == 0:
                    result['error'] = True
                    result['hasInfiniteSolutions'] = True
                    result['solutionFailed'] = True
                    result['errorMessage'] = 'The system has infinite solutions'
                    break
                else:
                    A, b = util.split_system(cp.deepcopy(ab))
                    step = { 'aMatrix': cp.deepcopy(A), 'bMatrix': cp.deepcopy(b) }
                    sorting_iterations.append(cp.deepcopy(step))

            for i in range(k, n):
                multiplier = A[i][k-1] / A[k-1][k-1]

                if np.isnan(multiplier) or A[k-1][k-1] == 0:
                    raise ZeroDivisionError

                for j in range(k, n+1):
                    A[i][j-1] = np.float64(A[i][j-1] - multiplier*A[k-1][j-1])

                b[i] = np.float64(b[i] - multiplier*b[k-1])

            if k < n - 1:
                iteration = { 'aMatrix': cp.deepcopy(A), 'bMatrix': cp.deepcopy(b) }
                iterations.append(cp.deepcopy(iteration))

        if not result['error']:
            result['aMatrix'] = A
            result['bMatrix'] = b
            result['xMatrix'] = util.regressive_substitution(A, b)
            result['sortingIterations'] = sorting_iterations
            result['iterations'] = iterations

        return result

    def partial_pivoting(self, Ab, n, k):
        upper = abs(Ab[k][k])
        upper_row = k

        for s in range(k+1, n):
            if abs(Ab[s][k]) > upper:
                upper = abs(Ab[s][k])
                upper_row = s
        if upper == 0:
            return []
        else:
            if upper_row != k:
                Ab = util.change_rows(Ab, upper_row, k)

            return Ab

    def complete_pivoting(self, Ab, n, k, marks):
        upper = 0
        upper_row = k
        upper_column = k

        for r in range(k, n):
            for s in range(k, n):
                if abs(Ab[r][s]) > upper:
                    upper = abs(Ab[r][s])
                    upper_row = r
                    upper_column = s

        if upper == 0:
            return []
        else:
            if upper_row != k:
                Ab = util.change_rows(Ab, upper_row,k)
            if upper_column != k:
                Ab = util.change_columns(Ab, upper_column, k)
                marks = util.change_marks(marks, upper_column, k)

            return Ab
