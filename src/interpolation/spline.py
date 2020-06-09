import copy as cp
import numpy as np

class Spline():
    def cubic_spline(self, points):
        result = {
            'aMatrix': None,
            'bMatrix': None,
            'resultMessage': None,
            'error': False,
            'errorMessage': None
        }
        n = len(points)

        if n <= 2:
            result['error'] = True
            result['errorMessage']= 'Points must be greater than 2'
        else:
            A = list()
            b = list()
            eq1 = n-1
            k = 0

            while k < eq1:
                for i in range(2):
                    ec = []

                    for j in range(eq1):
                        if j == k:
                            if i == 0:
                                x = [points[k][0]**3, points[k][0]**2, points[k][0], 1]
                            else:
                                x = [points[k+1][0]**3, points[k+1][0]**2, points[k+1][0], 1]
                            ec += x
                        else:
                            x = [0, 0, 0, 0]
                            ec += x

                    A.append(ec)
                    b.append(points[k+i][1])

                k += 1

            eq2 = n-2
            k = 0

            while k < eq2:
                der1 = []
                der2 = []

                for j in range(eq1):
                    if j == k:
                        x = [3*points[k+1][0]**2, 2*points[k+1][0], 1, 0]
                        der1 += x
                        x = [6*points[k+1][0], 2, 0, 0]
                        der2 += x
                    elif j== k+1:
                        x = [-3*points[k+1][0]**2, -2*points[k+1][0], -1, 0]
                        der1 += x
                        x = [-6*points[k+1][0], -2, 0, 0]
                        der2 += x
                    else:
                        x = [0, 0, 0, 0]
                        der1 += x
                        der2 += x

                A.append(der1)
                A.append(der2)
                b.append(0)
                b.append(0)
                k += 1

            x = [6*points[0][0], 2, 0, 0]

            for j in range(eq1-1):
                x += [0, 0, 0, 0]

            A.append(x)
            b.append(0)
            x = []

            for j in range(eq1-1):
                x += [0, 0, 0, 0]

            x += [6*points[n-1][0], 2, 0, 0]
            A.append(x)
            b.append(0)

        if not result['error']:
            result['aMatrix'] = A
            result['bMatrix'] = b
            result['resultMesage'] = 'The method successfully created the system of equations'

        return result
