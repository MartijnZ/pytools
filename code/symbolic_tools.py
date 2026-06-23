from importlib import resources
import sympy as sp
import numpy as np


class EquationEvaluator(object):

    def __init__(self, equation_string):
        """ Transforms a string containing an equation executable.
        :param equation_string: Equation string (e.g. 'm*c**2')
        """
        self._expr_str = equation_string

    def eval_expr(self, var_dict):
        """ Returns the evaluation of the equation for the variables defined in the var_dict
        :param var_dict: Dictionary of variables defined in the EquationEvaluator equation (e.g. {'m': 10, 'c': 300e6} for m*c**2)
        """
        # Make values available in local workspace:
        for name in var_dict.keys():
            exec('{0} = var_dict[\'{0}\']'.format(name))

        # Expose common items
        cos = np.cos
        sin = np.sin
        tan = np.tan
        Matrix = sp.Matrix

        # Execute expression:
        return eval('{0}'.format(self.equation))

    @property
    def equation(self):
        """Returns the equation defined for this EquationEvaluator object."""
        return self._expr_str

    def __str(self):
        return 'Equation Evaluator ({0})'.format(self.equation)

    @staticmethod
    def load(filename):
        """ Load symbolic expression from string file

        :param filename:
        :return:
        """
        with open(filename, 'r') as f:
            return EquationEvaluator(f.read())

    def save(self, filename):
        """ Save symbolic expression to string file

        :param filename:
        :return:
        """
        with open(filename, 'w') as f:
            f.write(self._expr_str)


def get_module_equations():
    equations_dir = resources.files("madernpytools").joinpath("data", "equations")

    eq_dict = {}
    for resource in equations_dir.iterdir():
        if not resource.is_file():
            continue

        eq_dict[resource.stem] = EquationEvaluator.load(resource)

    return eq_dict
