class Integration:
    def trapeze(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        eq = ''
        h = points[-1][0] - points[0][0]
        h = round(h,2)
        eq += f'({h}/2)'
        incr = points[0][1] + points[-1][1]
        eq += f'(({points[0][1]}) + ({points[-1][1]}))'
        res = (h * incr) / 2

        if result['error'] == False:
            result['resultMessage'] = 'The method with 2 points was successfull'
            result['values']['equation'] = eq
            result['values']['functionOutput'] = res

        return result

    def G_trapeze(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        equation_list = list()
        n = len(points)
        h = points[1][0] - points[0][0]
        h = round(h,2)
        eq = f'({h}/2)'
        incr = points[0][1]  +  points[-1][1]
        eq += f'(({points[0][1]}) + ({points[-1][1]}) 2('

        for i in range(1, n - 1):
            incr += 2 * points[i][1]
            eq += f'({points[i][1]}) +' if i < n - 2 else f'({points[i][1]}))'
            equation_list.append(eq)
            eq = ''

        res = (h * incr) / 2

        if result['error'] == False:
            result['resultMessage'] = 'The generalized method with 2 points was successfull'
            result['values']['equation'] = equation_list
            result['values']['functionOutput'] = res

        return result

    def simpson1_3(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        n = len(points)
        eq = ''
        res = 0

        if (n - 1) % 2 != 0:
            result['error'] = True
            result['errorMessage']= 'Points must be even'
        else:
            medio = int((n - 1) / 2)
            h = points[medio][0] - points[0][0]
            h = round(h,2)
            eq += f'({h}/3)'
            incr = points[0][1] + 4 * points[medio][1] + points[-1][1]
            eq += f'(({points[0][1]}) + 4({points[medio][1]}) + ({points[-1][1]}))'
            res = (h * incr) / 3

        if result['error'] == False:
            result['resultMessage'] = 'The method with 3 points was successfull'
            result['values']['equation'] = eq
            result['values']['functionOutput'] = res

        return result

    def simpson1_3G(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        n = len(points)
        equation_list = list()
        res = 0

        if (n - 1) % 2 != 0:
            result['error'] = True
            result['errorMessage']= 'Points must be even'
        else:
            h = points[1][0] - points[0][0]
            h = round(h,2)
            eq = f'({h}/3)'
            incr = points[0][1]  +  points[-1][1]
            eq += f'(({points[0][1]} + ({points[-1][1]}) + 2('

            for i in range(1, n - 1):
                if i % 2 == 0:
                    incr += 2 * points[i][1]
                    eq += f'({points[i][1]}) +'  if i < n - 3 else f'({points[i][1]})) +'
                    equation_list.append(eq)
                    eq = ''

            eq += f'4('

            for i in range(1, n - 1):
                if i % 2 != 0:
                    incr += 4 * points[i][1]
                    eq += f'({points[i][1]}) +' if i < n - 2 else f'({points[i][1]}))'
                    equation_list.append(eq)
                    eq = ''

            res = (h * incr) / 3

        if result['error'] == False:
            result['resultMessage'] = 'The generalized method with 3 points was successfull'
            result['values']['equation'] = equation_list
            result['values']['functionOutput'] = res

        return result

    def simpson3_8(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        n = len(points)
        eq = ''
        res = 0

        if (n - 1) % 3 != 0:
            result['error'] = True
            result['errorMessage']= 'Points must be multiple of 3'
        else:
            f1 = int((n - 1) / 3)
            f2 = 2 * int((n - 1) / 3)
            h = points[f1][0] - points[0][0]
            h = round(h,2)
            eq += f'(3*{h}/8)'
            incr = points[0][1] + 3 * points[f1][1] + 3 * points[f2][1] + points[-1][1]
            eq += f'(({points[0][1]}) + 3({points[f1][1]}) + 3({points[f2][1]}) + ({points[-1][1]}))'
            res = (3 * h * incr) / 8

        if result['error'] == False:
            result['resultMessage'] = 'The method with 4 points was successfull'
            result['values']['equation'] = eq
            result['values']['functionOutput'] = res

        return result

    def simpson3_8G(self, points):
        result = {
            'error': False,
            'errorMessage': None,
            'resultMessage': None,
            'solutionFailed': False,
            'values': {
                'equation': None,
                'functionOutput': None
            }
        }
        n = len(points)
        equation_list = list()
        res = 0

        if (n - 1) %3 != 0:
            result['error'] = True
            result['errorMessage']= 'Points must be multiple of 3'
        else:
            h = points[1][0] - points[0][0]
            h = round(h,2)
            eq = f'(3*{h}/8)'
            incr = points[0][1]  +  points[-1][1]
            eq += f'(({points[0][1]} + ({points[-1][1]}) + 2('

            for i in range(1, n - 1):
                if i%3 == 0:
                    incr += 2 * points[i][1]
                    eq += f'({points[i][1]}) +' if i < n - 4 else f'({points[i][1]})) +'
                    equation_list.append(eq)
                    eq = ''

            eq += f'3('

            for i in range(1, n - 1):
                if i%3 != 0:
                    incr += 3 * points[i][1]
                    eq += f'({points[i][1]}) +' if i < n - 2 else f'({points[i][1]}))'
                    equation_list.append(eq)
                    eq = ''

            res = (3 * h * incr) / 8

        if result['error'] == False:
            result['resultMessage'] = 'The generalized method with 4 points was successfull'
            result['values']['equation'] = equation_list
            result['values']['functionOutput'] = res

        return result
