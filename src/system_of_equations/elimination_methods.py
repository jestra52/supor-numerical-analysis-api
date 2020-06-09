class EliminationMethods:
    def __init__(self, matrix):
        self.matrix = matrix

    def dotMatrizWithMulplier(matrix_temp):
        if matrix_temp['index'] > matrix_temp['actual_index'] :
            matrix_temp['data'] = matrix_temp['data']- np.dot(matrix_temp['data'][matrix_temp['actual_index']]/matrix_temp['actual_pivot'][matrix_temp['actual_index']], matrix_temp['actual_pivot'])

        return matrix_temp

    def gaussian_elimination_method():
        if __name__ == "__main__":
            values = [{'data':x, 'index':i, 'actual_pivot' : [], 'actual_index': 0} for i,x in enumerate(self.matrix)]
            for i in range(0, len(values)-1):
                for j in range(0, len(values)):
                    values[j]['actual_index'] = i
                    values[j]['actual_pivot'] = values[i]['data']

                pool = multiprocessing.cpu_count()
                with multiprocessing.Pool(pool) as p:
                    values = list(p.map(dotList,values))
        return values

    def gaussian_elimination_method_with_simple_pivot():
        if __name__ == "__main__":
            values = [{'data':x, 'index':i, 'actual_pivot' : [], 'actual_index': 0} for i,x in enumerate(self.matrix)]
            for i in range(0, len(values)-1):
                max_pivot = abs(values[i]['datos'][i])
                for j in range(0, len(values)):
                    if max_pivot < abs(values[j]['datos'][j]):
                        max_pivot = abs(values[j]['datos'][j])
                if max_pivot != i:
                    temp_values = values[i]
                    values[i] = values[max_pivot]
                    values[max_pivot] = temp_values
                values[j]['actual_index'] =temp_values
                values[j]['actual_pivot'] = values[temp_values]['data']
                pool = multiprocessing.cpu_count()
                with multiprocessing.Pool(pool) as p:
                    values = list(p.map(dotList,values))
        return values