class Lagrange():
    def lagrange(self, points, x):
        result = {
            'error': False,
            'errorMessage': None,
            'functionOutput': None,
            'terms': None,
            'polynomial': None,
            'resultMessage': None,
            'solutionFailed': False
        }
        n = len(points)
        l_values = list()
        p_values = list()
        res = 0

        for i in range(n):
            num = 1
            den = 1
            pn = ''
            ln = '('

            for j in range(n):
                if j != i:
                    num *= (x - points[j][0])
                    ln += f'({x}-{points[j][0]})'

            ln += ') / ('

            for j in range(n):
                if j != i:
                    den *= (points[i][0] - points[j][0])
                    ln += f'({points[i][0]}-{points[j][0]})'

            ln += ')'

            if den == 0:
                raise ZeroDivisionError

            Lx = num / den
            ln += f' = {Lx}' if Lx != -0 else ' = 0.0'
            pn += f'({Lx})({points[i][1]})' if Lx != -0 else f'(0.0)({points[i][1]})'
            pn += ' +' if i < n - 1 else ''

            LxFx = Lx * points[i][1]
            res += LxFx

            l_values.append(ln)
            p_values.append(pn)

        result['terms'] = l_values
        result['polynomial'] = p_values
        result['functionOutput'] = res

        return result
