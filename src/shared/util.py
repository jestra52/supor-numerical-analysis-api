import math as mt
import numpy as np

messageType = {
    'ERROR': 'error',
    'INFO': 'info',
    'SUCCESS': 'success',
    'WARNING': 'warning'
}

responseMessage = {
    'ERROR': 'There was an error executing the method',
    'SUCCESS': 'The method was successfully executed',
    'ZERO_DIVISION_ERROR': 'There is a division by zero'
}

method_type = {
    'FACT_CHOLESKY': 'cholesky',
    'FACT_CROUT': 'crout',
    'FACT_DOOLITTLE': 'doolittle',
    'GAUSS_COMPLETE': 'complete',
    'GAUSS_PARTIAL': 'partial',
    'GAUSS_SIMPLE': 'simple',
    'ITER_GAUSS': 'gauss',
    'ITER_JACOBI': 'jacobi',
    'INTERPOLATION_NEWTON': 'intNewton',
    'INTERPOLATION_LAGRANGE': 'intLagrange',
    'INTEGRATION_SIMPLE': 'integrationSimple',
    'INTEGRATION_GENERAL': 'integrationGeneral',
    'INTEGRATION_TRAPEZE': 'integrationTrapeze',
    'INTEGRATION_SIMPSON13': 'simpson13',
    'INTEGRATION_SIMPSON38': 'simpson38'
}

def getClosedIntermediateValue(xi, xs, fxi, fxs, is_false_rule):
    return xi - fxi*(xs-xi)/(fxs-fxi) if is_false_rule else (xi+xs)/2

def getError(current_x, prev_x, is_relative_error):
    return np.abs((current_x-prev_x)/current_x) if is_relative_error else np.abs(current_x-prev_x)

def getMatrixError(is_relative_error, xold, xnew):
        incr = 0
        if not is_relative_error:

            for i in range(len(xold)):
                incr += (xnew[i]-xold[i])**2

            error = mt.sqrt(incr)
        else:
            den = 0

            for i in range(len(xold)):
                incr+=(xnew[i]-xold[i])**2
                den+=xnew[i]**2

            if den == 0:
                raise ZeroDivisionError

            error = mt.sqrt(incr) / mt.sqrt(den)
        return error

def regressive_substitution(A, b):
    n = len(A)
    x = np.zeros(n)

    for i in range(n, 0, -1):
        if np.isnan(b[i-1]) or A[i-1][i-1] == 0:
            raise ZeroDivisionError

        incr = 0

        for p in range(i + 1, n + 1, 1):
            if np.isnan(A[i-1][p-1]):
                raise ZeroDivisionError

            incr += np.float64(A[i-1][p-1] * x[p-1])

        x[i-1] =  np.float64((b[i-1] - incr) / A[i-1][i-1])

    return x

def augmented_matrix(A, b):
    for i in range(0, len(A)):
        A[i].append(b[i])

    return A

def split_system(Ab):
    b = []

    for i in range(0, len(Ab)):
        b.append(Ab[i][len(Ab)])
        Ab[i].pop()

    return (Ab, b)

def change_rows(A, upperRow, k):
    for i in range(len(A[0])):
        aux = A[k][i]
        A[k][i] = A[upperRow][i]
        A[upperRow][i] = aux

    return A

def change_columns(A, upper_column, k):
    for i in range(len(A[0])-1):
        aux = A[i][k]
        A[i][k] = A[i][upper_column]
        A[i][upper_column] = aux

    return A

def change_marks(marks, upper_column, k):
    aux = marks[upper_column]
    marks[upper_column] = marks[k]
    marks[k] = aux

    return marks

def parse_float_matrix(matrix):
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            matrix[i][j] = np.float64(matrix[i][j])

    return matrix

def parse_float_array(array):
    for i in range(0, len(array)):
        array[i] = np.float64(array[i])

    return array
