class NewtonInterpolation():
    def newton(self, points, x):
        result = {
            'terms': None,
            'error': False,
            'errorMessage': None,
            'functionOutput': None,
            'polynomial': None,
            'resultMessage': None,
            'solutionFailed': False
        }
        n = len(points)
        res = 0
        b_values = list()
        p_values = list()
        x_b = []

        for i in range(n):
            pn = ''
            x_b.append(points[i])
            b, form = self.solve_b(x_b)
            b_values.append(f'{form} = {str(b)}')
            pn += str(b) if b > 0 else f'({str(b)})'
            prod = 1

            for j in range(i):
                prod = prod * (x-points[j][0])
                pn += f'(x-{points[j][0]})'

            pn += ' +' if i < n-1 else ''
            tx = b * prod
            res += tx

            p_values.append(pn)

        result['terms'] = b_values
        result['functionOutput'] = res
        result['polynomial'] = p_values

        return result

    def solve_b(self, x_b):
        if len(x_b) == 1:
            return x_b[0][1],x_b[0][1]
        else:
            form = ''
            den = x_b[0][0] - x_b[-1][0]

            if den == 0:
                raise ZeroDivisionError

            b1 = self.solve_b(x_b[:-1])
            b2 = self.solve_b(x_b[1:])
            b = (b1[0] - b2[0]) / den
            form = f'({b1[0]} - {b2[0]}) / {den}'

            return b, form
