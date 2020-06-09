from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from interpolation import NewtonInterpolation, Lagrange, Spline
from numerical_integration import Integration
from one_variable_methods import ClosedMethods, OpenMethods
from shared import util
from shared.function_manager import FunctionManager
from system_of_equations import GaussElimination, Factorization, IterativeMethods
import numpy as np
import pandas as pd
import simplejson

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

closed_methods = ClosedMethods('x')
open_methods = OpenMethods('x', 'x')
gauss_elimination = GaussElimination()
factorization = Factorization()
iterative_methods = IterativeMethods()
newton_interpolation = NewtonInterpolation()
lagrange_interpolation = Lagrange()
spline_interpolation = Spline()
integration = Integration()

@app.route('/api/closedMethods/incrSearch', methods=['POST'])
@cross_origin()
def get_incr_search_result():
    dto = {
        'data': {
            'aproxValue': { 'x0': None, 'x1': None },
            'isExact': False,
            'n': 0,
            'result': {
                'x': None,
                'fx': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        closed_methods.set_fx(request.json.get('fx'))
        methodResult = closed_methods.incr_search(
            x0=np.float64(request.json.get('x0')),
            delta=np.float64(request.json.get('delta')),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'aproxValue': methodResult.get('aproxValue'),
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'x': methodResult.get('values').get('x').tolist(),
                    'fx': methodResult.get('values').get('fx').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/closedMethods/bisecFalseRule', methods=['POST'])
@cross_origin()
def get_bisec_false_rule_result():
    dto = {
        'data': {
            'aproxValue': { 'x0': None, 'x1': None },
            'isExact': False,
            'n': 0,
            'result': {
                'error': None,
                'fxm': None,
                'xi': None,
                'xm': None,
                'xs': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        closed_methods.set_fx(request.json.get('fx'))
        methodResult = closed_methods.bisec_false_rule(
            xi=np.float64(request.json.get('xi')),
            xs=np.float64(request.json.get('xs')),
            tol=np.float64(request.json.get('tol')),
            is_rel_error=request.json.get('isRelError'),
            is_false_rule=request.json.get('isFalseRule'),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'aproxValue': methodResult.get('aproxValue'),
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'xi': methodResult.get('values').get('xi').tolist(),
                    'xs': methodResult.get('values').get('xs').tolist(),
                    'xm': methodResult.get('values').get('xm').tolist(),
                    'fxm': methodResult.get('values').get('fxm').tolist(),
                    'error': methodResult.get('values').get('error').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/openMethods/fixedPoint', methods=['POST'])
@cross_origin()
def get_fixed_point_result():
    dto = {
        'data': {
            'isExact': False,
            'n': 0,
            'result': {
                'error': None,
                'fxn': None,
                'xn': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        open_methods.set_fx(request.json.get('fx'))
        open_methods.set_gx(request.json.get('gx'))
        methodResult = open_methods.fixed_point_iteration(
            xa=np.float64(request.json.get('x0')),
            tol=np.float64(request.json.get('tol')),
            is_rel_error=request.json.get('isRelError'),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'xn': methodResult.get('values').get('xn').tolist(),
                    'fxn': methodResult.get('values').get('fxn').tolist(),
                    'error': methodResult.get('values').get('error').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/openMethods/newton', methods=['POST'])
@cross_origin()
def get_newton_result():
    dto = {
        'data': {
            'isExact': False,
            'n': 0,
            'result': {
                'error': None,
                'fxn': None,
                'dfxn': None,
                'xn': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        open_methods.set_fx(request.json.get('fx'))
        methodResult = open_methods.newton(
            x0=np.float64(request.json.get('x0')),
            tol=np.float64(request.json.get('tol')),
            is_rel_error=request.json.get('isRelError'),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'xn': methodResult.get('values').get('xn').tolist(),
                    'fxn': methodResult.get('values').get('fxn').tolist(),
                    'dfxn': methodResult.get('values').get('dfxn').tolist(),
                    'error': methodResult.get('values').get('error').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/openMethods/secant', methods=['POST'])
@cross_origin()
def get_secant_result():
    dto = {
        'data': {
            'isExact': False,
            'n': 0,
            'result': {
                'den': None,
                'error': None,
                'fxn': None,
                'xn': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        open_methods.set_fx(request.json.get('fx'))
        methodResult = open_methods.secant(
            x0=np.float64(request.json.get('x0')),
            x1=np.float64(request.json.get('x1')),
            tol=np.float64(request.json.get('tol')),
            is_rel_error=request.json.get('isRelError'),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'xn': methodResult.get('values').get('xn').tolist(),
                    'fxn': methodResult.get('values').get('fxn').tolist(),
                    'den': methodResult.get('values').get('den').tolist(),
                    'error': methodResult.get('values').get('error').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/openMethods/multipleRoots', methods=['POST'])
@cross_origin()
def get_multiple_roots_result():
    dto = {
        'data': {
            'isExact': False,
            'n': 0,
            'result': {
                'd2fxn': None,
                'dfxn': None,
                'error': None,
                'fxn': None,
                'xn': None
            },
            'resultMessage': None,
            'root': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None

    try:
        open_methods.set_fx(request.json.get('fx'))
        methodResult = open_methods.multiple_roots(
            x0=np.float64(request.json.get('x0')),
            tol=np.float64(request.json.get('tol')),
            is_rel_error=request.json.get('isRelError'),
            n=int(request.json.get('n')))

        if methodResult.get('error') == False:
            dto['data'] = {
                'isExact': methodResult.get('isExact'),
                'n': methodResult.get('n'),
                'result': {
                    'xn': methodResult.get('values').get('xn').tolist(),
                    'fxn': methodResult.get('values').get('fxn').tolist(),
                    'dfxn': methodResult.get('values').get('dfxn').tolist(),
                    'd2fxn': methodResult.get('values').get('d2fxn').tolist(),
                    'error': methodResult.get('values').get('error').tolist()
                },
                'resultMessage': methodResult.get('resultMessage'),
                'root': methodResult.get('root'),
                'solutionFailed': methodResult.get('solutionFailed')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = methodResult.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/sysOfLinEquations/gaussElimination', methods=['POST'])
@cross_origin()
def get_gauss_elimination_result():
    dto = {
        'data': {
            'result': {
                'aMatrix': None,
                'bMatrix': None,
                'xMatrix': None,
                'iterations': None
            },
            'hasInfiniteSolutions': None,
            "solutionFailed": False,
            'resultMessage': None
        },
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        method_result = gauss_elimination.execute(
            A=util.parse_float_matrix(request.json.get('aMatrix')),
            b=util.parse_float_array(request.json.get('bMatrix')),
            method_type=request.json.get('methodType'))

        if method_result.get('error') == False:
            dto['data'] = {
                'result': {
                    'aMatrix': list(map(lambda a: list(a), method_result.get('aMatrix'))),
                    'bMatrix': list(method_result.get('bMatrix')),
                    'xMatrix': list(method_result.get('xMatrix')),
                    'iterations': method_result.get('iterations'),
                    'sortingIterations': method_result.get('sortingIterations')
                },
                'hasInfiniteSolutions': method_result.get('hasInfiniteSolutions'),
                "solutionFailed": method_result.get('hasInfiniteSolutions'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = f'{util.responseMessage.get("ZERO_DIVISION_ERROR")}: the system may have infinite solutions or no solution'
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/sysOfLinEquations/factorization', methods=['POST'])
@cross_origin()
def get_factorization_result():
    dto = {
        'data': {
            'result': {
                'aMatrix': None,
                'bMatrix': None,
                'lMatrix': None,
                'uMatrix': None,
                'xMatrix': None,
                'iterations': None
            },
            'hasInfiniteSolutions': None,
            "solutionFailed": False,
            'resultMessage': None
        },
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        A = util.parse_float_matrix(request.json.get('aMatrix'))
        b = util.parse_float_array(request.json.get('bMatrix'))

        if request.json.get('methodType') == util.method_type.get('FACT_DOOLITTLE'):
            method_result = factorization.doolittle(A,b)
        elif request.json.get('methodType') == util.method_type.get('FACT_CROUT'):
            method_result = factorization.crout(A,b)
        elif request.json.get('methodType') == util.method_type.get('FACT_CHOLESKY'):
            method_result = factorization.cholesky(A,b)

        if method_result.get('error') == False:
            dto['data'] = {
                'result': {
                    'aMatrix': list(map(lambda a: list(a), method_result.get('aMatrix'))),
                    'lMatrix': list(map(lambda l: list(l), method_result.get('lMatrix'))),
                    'uMatrix': list(map(lambda u: list(u), method_result.get('uMatrix'))),
                    'bMatrix': list(method_result.get('bMatrix')),
                    'xMatrix': list(method_result.get('xMatrix')),
                    'iterations': method_result.get('iterations')
                },
                'hasInfiniteSolutions': method_result.get('hasInfiniteSolutions'),
                "solutionFailed": method_result.get('hasInfiniteSolutions'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = f'{util.responseMessage.get("ZERO_DIVISION_ERROR")}: the system may have infinite solutions or no solution'
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (ValueError) as ex:
        dto['message'] = f'There is a negative number inside a square root. Please check your system'
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/sysOfLinEquations/iterativeMethods', methods=['POST'])
@cross_origin()
def get_iterative_method_result():
    dto = {
        'data': {
            'n': 0,
            'result': {
                'error': None,
                'x': None
            },
            'resultMessage': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        if request.json.get('methodType') == util.method_type.get('ITER_GAUSS'):
            method_result = iterative_methods.gauss_seidel(
                A=util.parse_float_matrix(request.json.get('aMatrix')),
                b=util.parse_float_array(request.json.get('bMatrix')),
                iterations=np.float64(request.json.get('n')),
                tol=np.float64(request.json.get('tol')),
                l=np.float64(request.json.get('l')),
                xi=util.parse_float_array(request.json.get('xArray')),
                is_rel_error=request.json.get('isRelError'))
        elif request.json.get('methodType') == util.method_type.get('ITER_JACOBI'):
            method_result = iterative_methods.jacobi(
                A=util.parse_float_matrix(request.json.get('aMatrix')),
                b=util.parse_float_array(request.json.get('bMatrix')),
                iterations=np.float64(request.json.get('n')),
                tol=np.float64(request.json.get('tol')),
                l=np.float64(request.json.get('l')),
                x=util.parse_float_array(request.json.get('xArray')),
                is_rel_error=request.json.get('isRelError'))

        if method_result.get('error') == False:
            x = util.parse_float_array(request.json.get('xArray'))
            x_results = list()

            for i in range(len(x)):
                x_results.append(method_result.get('values').get(f'x{i}').tolist())

            dto['data'] = {
                'result': {
                    'n': method_result.get('n'),
                    'error': method_result.get('values').get('error').tolist(),
                    'x': x_results
                },
                'solutionFailed': method_result.get('hasInfiniteSolutions'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/sysOfLinEquations/getBestLambda', methods=['POST'])
@cross_origin()
def get_iterative_method_best_lambda_result():
    dto = {
        'data': { 'lambda': None, 'resultMessage': None },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        if request.json.get('methodType') == util.method_type.get('ITER_GAUSS'):
            method_result = iterative_methods.best_lambda_g(
                A=util.parse_float_matrix(request.json.get('aMatrix')),
                b=util.parse_float_array(request.json.get('bMatrix')),
                x=util.parse_float_array(request.json.get('xArray')),
                is_rel_error=request.json.get('isRelError'))
        elif request.json.get('methodType') == util.method_type.get('ITER_JACOBI'):
            method_result = iterative_methods.best_lambda_j(
                A=util.parse_float_matrix(request.json.get('aMatrix')),
                b=util.parse_float_array(request.json.get('bMatrix')),
                x=util.parse_float_array(request.json.get('xArray')),
                is_rel_error=request.json.get('isRelError'))

        if method_result.get('error') == False:
            dto['data'] = {
                'lambda': method_result.get('lambda'),
                'resultMessage': method_result.get('resultMessage')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/interpolation/newtonLagrange', methods=['POST'])
@cross_origin()
def get_interpolation_newton_lagrange_result():
    dto = {
        'data': {
            'result': {
                'functionOutput': None,
                'terms': None,
                'polynomial': None
            },
            'resultMessage': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        if request.json.get('methodType') == util.method_type.get('INTERPOLATION_NEWTON'):
            method_result = newton_interpolation.newton(
                points=util.parse_float_matrix(request.json.get('points')),
                x=np.float64(request.json.get('x')))
        elif request.json.get('methodType') == util.method_type.get('INTERPOLATION_LAGRANGE'):
            method_result = lagrange_interpolation.lagrange(
                points=util.parse_float_matrix(request.json.get('points')),
                x=np.float64(request.json.get('x')))

        if method_result.get('error') == False:
            dto['data'] = {
                'result': {
                    'functionOutput': method_result.get('functionOutput'),
                    'polynomial': method_result.get('polynomial'),
                    'terms': method_result.get('terms')
                },
                'solutionFailed': method_result.get('solutionFailed'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/interpolation/splines', methods=['POST'])
@cross_origin()
def get_interpolation_splines_result():
    dto = {
        'data': {
            'result': {
                'aMatrix': None,
                'bMatrix': None,
                'iterations': None,
                'xMatrix': None
            },
            'resultMessage': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        method_result = spline_interpolation.cubic_spline(
            points=util.parse_float_matrix(request.json.get('points')))

        if method_result.get('error') == False:
            method_solution = gauss_elimination.execute(
                A=util.parse_float_matrix(method_result.get('aMatrix')),
                b=util.parse_float_array(method_result.get('bMatrix')),
                method_type=util.method_type.get('GAUSS_COMPLETE'))
            dto['data'] = {
                'result': {
                    'aMatrix': method_result.get('aMatrix'),
                    'bMatrix': method_result.get('bMatrix'),
                    'xMatrix': list(method_solution.get('xMatrix'))
                },
                'solutionFailed': method_result.get('solutionFailed'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

@app.route('/api/integration', methods=['POST'])
@cross_origin()
def get_integration_result():
    dto = {
        'data': {
            'result': {
                'equation': None,
                'functionOutput': None
            },
            'resultMessage': None,
            'solutionFailed': False
        },
        'errorDetails': None,
        'message': None,
        'messageType': util.messageType.get('SUCCESS'),
        'success': True
    }
    response = None
    method_result = None

    try:
        points=util.parse_float_matrix(request.json.get('points'))

        if request.json.get('formType') == util.method_type.get('INTEGRATION_SIMPLE'):
            if request.json.get('methodType') == util.method_type.get('INTEGRATION_TRAPEZE'):
                method_result = integration.trapeze(points)
            elif request.json.get('methodType') == util.method_type.get('INTEGRATION_SIMPSON13'):
                method_result = integration.simpson1_3(points)
            elif request.json.get('methodType') == util.method_type.get('INTEGRATION_SIMPSON38'):
                method_result = integration.simpson3_8(points)
        if request.json.get('formType') == util.method_type.get('INTEGRATION_GENERAL'):
            if request.json.get('methodType') == util.method_type.get('INTEGRATION_TRAPEZE'):
                method_result = integration.G_trapeze(points)
            elif request.json.get('methodType') == util.method_type.get('INTEGRATION_SIMPSON13'):
                method_result = integration.simpson1_3G(points)
            elif request.json.get('methodType') == util.method_type.get('INTEGRATION_SIMPSON38'):
                method_result = integration.simpson3_8G(points)

        if method_result.get('error') == False:
            dto['data'] = {
                'result': {
                    'equation': method_result.get('values').get('equation'),
                    'functionOutput': method_result.get('values').get('functionOutput')
                },
                'solutionFailed': method_result.get('solutionFailed'),
                'resultMessage': util.responseMessage.get('SUCCESS')
            }
            dto['message'] = util.responseMessage.get('SUCCESS')
            response = make_response(jsonify(dto), 200)
        else:
            dto['message'] = method_result.get('errorMessage')
            dto['messageType'] = util.messageType.get('ERROR')
            response = make_response(jsonify(dto), 400)
    except (ZeroDivisionError) as ex:
        dto['message'] = util.responseMessage.get('ZERO_DIVISION_ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 400)
    except (Exception, RuntimeError, TypeError) as ex:
        dto['message'] = util.responseMessage.get('ERROR')
        dto['errorDetails'] = str(ex)
        dto['messageType'] = util.messageType.get('ERROR')
        response = make_response(jsonify(dto), 500)

    return response

if __name__ == '__main__':
    app.run(debug=True)
