import traitlets, re
import numpy as np
import sympy as sp
import madernpytools.tools.symbolic_tools as msym
from matplotlib import lines


class GraphicElement(object):

    @property
    def value_dict(self) -> dict:
        """ Returns a dictionary of symbol names (key) and their current corresponding values
        """
        raise NotImplementedError()

    @property
    def sub_elements(self) -> dict:
        """ Returns a dictionary of sub elements
        """
        raise NotImplementedError()

    @property
    def symbols(self) -> dict:
        """ Returns a dictionary of symbols
        """
        raise NotImplementedError()

    def draw(self, ax):
        raise NotImplementedError()

    def symbol2attrname(self, symbol) -> str:
        """Returns the attribute name that corresponds to the provided symbol. Returns None if object does not contain the attribute
        """
        raise NotImplementedError()

    def __contains__(self, item):
        """ Checks if item (symbol, or attribute) is contained in Graphic element

        @param item:
        @return:
        """
        return hasattr(self, str(item)) or str(item) in [str(s) for _, s in self.symbols.items()]

    def __setitem__(self, key, value):
        """ Assign value to key attribute
        If key matches an object symbolic name, it is mapped to the corresponding attribute
        If key mathces an object (symbolic name) of a sub element, it is mapped ot the sub-element.

        @param key:
        @param value:
        @return:
        """
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self, self.symbol2attrname(key)):
            self[self.symbol2attrname(key)] = value
        else:
            for _, elem in self.sub_elements.items():
                if key in elem:
                    elem[key] = value

    def __getitem__(self, key):
        """ Assign value to key attribute
        If key matches an object symbolic name, it is mapped to the corresponding attribute
        If key mathces an object (symbolic name) of a sub element, it is mapped ot the sub-element.

        @param key:
        @return:
        """
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, self.symbol2attrname(key)):
            return self[self.symbol2attrname(key)]
        else:
            for _, elem in self.sub_elements.items():
                if key in elem:
                    return elem[key]

    def __str__(self):
        return f'Graphic element (id(self))'


class Point2D(traitlets.HasTraits, traitlets.TraitType, GraphicElement):
    x = traitlets.CFloat(default_value=0.0)
    y = traitlets.CFloat(default_value=0.0)

    def __init__(self, x: float = 0.0, y: float = 0.0, name='', **kwargs):

        if name == '':
            self._symbs = dict(zip(['x', 'y'], sp.symbols(f'x_{id(self)} y_{id(self)}')))
        else:
            self._symbs = dict(zip(['x', 'y'], sp.symbols(f'x_{name} y_{name}')))

        #  Display item:
        self.plot_element = lines.Line2D([x], [y], marker='.', markersize=10, **kwargs)
        self._name = name

        super().__init__(x=x, y=y)

    def symbol2attrname(self, item):
        if 'x' in str(item):
            return 'x'
        elif 'y' in str(item):
            return 'y'

    @traitlets.observe('x', 'y')
    def _coordinate_change(self, change):
        self.plot_element.set_data([self.x], [self.y])

    @property
    def sub_elements(self):
        return {}

    @property
    def symbols(self):
        return self._symbs

    @property
    def value_dict(self):
        return {str(self._symbs['x']): self.x, str(self._symbs['y']): self.y}

    def draw(self, ax):
        ax.add_line(self.plot_element)

    def __str__(self):
        return f"""Point (id: {id(self)}): x={self.x}, y={self.y} '
        """


class Line2D(traitlets.HasTraits, GraphicElement):
    p1 = Point2D()
    p2 = Point2D()

    def __init__(self, p1: Point2D, p2: Point2D, **kwargs):
        self.plot_element = lines.Line2D([], [], **kwargs)
        super().__init__(p1=p1, p2=p2)

    @property
    def sub_elements(self):
        return {'p1': self.p1, 'p2': self.p2}

    def symbol2attrname(self, key):
        for _, elem in self.sub_elements.items():
            if elem.symbol2attrname(key):
                return elem.symbol2attrname(key)

    @traitlets.observe('p1', 'p2')
    def _point_change(self, change):
        # Subscribe to point changes:
        if isinstance(change.old, Point2D):
            change.old.unobserve(self._update_points, ['x', 'y'])
        if isinstance(change.new, Point2D):
            change.new.observe(self._update_points, ['x', 'y'])

        # Update points:
        self._update_points(None)

    def _update_points(self, _):
        self.plot_element.set_data([self.p1.x, self.p2.x],
                                   [self.p1.y, self.p2.y])

    def draw(self, ax):
        self.p1.draw(ax)
        self.p2.draw(ax)
        ax.add_line(self.plot_element)

    @property
    def symbols(self):
        p1symbs = {f'p1_{key}': val for key, val in self.p1.symbols.items()}
        p2symbs = {f'p2_{key}': val for key, val in self.p2.symbols.items()}
        return {**p1symbs, **p2symbs}

    @property
    def value_dict(self):
        return {**self.p1.value_dict, **self.p2.value_dict}

    def __str__(self):
        return f"""Line2D ({id(self)}): 
            {str(self.p1)}
            {str(self.p2)}

        """


class Arc2D(GraphicElement, traitlets.HasTraits, traitlets.TraitType):
    p_c = Point2D()
    r = traitlets.CFloat(default_value=1.0)
    a_start = traitlets.CFloat(default_value=0.0)
    a_stop = traitlets.CFloat(default_value=3.14159 / 4)
    show_end_points = traitlets.CInt(default_value=0)

    def __init__(self, p_c, r, a_start, a_stop, show_end_points=0, **kwargs):
        self.plot_element = lines.Line2D([], [], **kwargs)
        self.plot_end_points = lines.Line2D([], [], ls='None', marker='.', **kwargs)
        super().__init__(p_c=p_c, r=r, a_start=a_start, a_stop=a_stop, show_end_points=show_end_points)

        self._symbs = dict(zip(['r', 'a_start', 'a_stop'],
                               sp.symbols(f'r_{id(self)} a_start_{id(self)} a_stop_{id(self)}')))

    @traitlets.observe('p_c')
    def _point_change(self, change):
        # Subscribe to point changes:
        if isinstance(change.old, Point2D):
            change.old.unobserve(self._update_points, ['x', 'y'])
        if isinstance(change.new, Point2D):
            change.new.observe(self._update_points, ['x', 'y'])

        # Update points:
        self._update_points(None)

    @property
    def sub_elements(self):
        return {'p_c': self.p_c}

    def symbol2attrname(self, key):
        for name in ['r', 'a_start', 'a_stop']:
            if re.match(f'{name}_\d*', key):
                return name

        for _, elem in self.sub_elements.items():
            if elem.symbol2attrname(key):
                return elem.symbol2attrname(key)

    @traitlets.observe('r', 'a_start', 'a_stop', 'show_end_points')
    def _update_points(self, _):
        angles = np.linspace(self.a_start, self.a_stop, 100)
        x_vals = self.p_c.x + self.r * np.cos(angles)
        y_vals = self.p_c.y + self.r * np.sin(angles)
        self.plot_element.set_data(x_vals,y_vals)

        if self.show_end_points == 0:
            self.plot_end_points.set_data([],[])
        elif self.show_end_points == 1:
            self.plot_end_points.set_data([x_vals[0]],[y_vals[0]])
        elif self.show_end_points == 2:
            self.plot_end_points.set_data([x_vals[-1]],[y_vals[-1]])
        elif self.show_end_points == 3:
            self.plot_end_points.set_data(x_vals[[0,-1]], y_vals[[0,-1]])

    def draw(self, ax):
        self.p_c.draw(ax)
        ax.add_line(self.plot_element)
        ax.add_line(self.plot_end_points)

    @property
    def symbols(self):
        return {**self._symbs, **self.p_c.symbols}

    @property
    def value_dict(self):
        return {**self.p_c.value_dict,
                str(self._symbs['r']): self.r,
                str(self._symbs['a_start']): self.a_start,
                str(self._symbs['a_stop']): self.a_stop}

    def __str__(self):
        return f"""Arc (id: {id(self)}):
        r       = {self.r} 
        a_start = {self.a_start} 
        a_stop  = {self.a_stop}  
        p_c     = {self.p_c} 
        """


class ConstraintSolver(object):

    def __init__(self):
        pass

    def solve(self, c_list: list, verbose=False):

        # Collect information from constraints:
        q_list = []  # Symbolic variables to use
        element_list = []  # List of graphic elements involved in optimization
        equation_list = []  # List of equations
        for c in c_list:

            # Add element if not yet in list:
            for e in c.elements:
                if e not in element_list:
                    element_list.append(e)

            # Add symbol if not yet in list:
            for q in c.symbols:
                if not q in q_list:
                    q_list.append(q)
            equation_list += c.equation

        # Collect initial guess:
        if verbose:
            print(f'Constraints: \n{equation_list}')
            print(f'Elements   : \n{element_list}')
            print(f'Q_list: \n{q_list}')

        q_init = {}
        for q in q_list:
            found = False
            for e in element_list:
                if str(q) in e:
                    tmp = e[str(q)]
                    if tmp is not None:
                        q_init[str(q)] = tmp
                    else:
                        raise RuntimeError(f'Did not find {q} in {e}')

                    found = True
                    break

            if not found:
                raise RuntimeError(f"""Could not find initial value for {q}
                in {" ".join([str(e) for e in element_list])} 
                """)

        # Run Newton-Rapshon:
        self._q = q_list
        self._c = sp.Matrix(equation_list)
        self._qinit = q_init

        val_dict, residue, iterations = self._newton_rapson()

        # Assign Optimal values:
        for key, value in val_dict.items():
            for e in element_list:
                if str(key) in e:
                    e[str(key)] = value
                    break

        return val_dict, residue, iterations

    def _newton_rapson(self):
        # Compute jacobian:
        Jac = sp.Matrix([[sp.diff(self._c, symbol) for symbol in self._q]])

        # Create evaluators to speed up computation:
        Jeval = msym.EquationEvaluator(str(Jac))
        ceval = msym.EquationEvaluator(str(self._c))

        # Iteratively solve:
        cnt = 0

        q_init = {}
        for key, val in self._qinit.items():
            q_init[
                key] = val + np.random.randn() * 0.01  # Add some perturbation to ensure all search directions are triggered

        iterations = [q_init]
        val_dict = q_init
        while True:
            cnt += 1
            vals = np.array([val for _, val in val_dict.items()])
            keys = val_dict.keys()
            val_dict['acos'] = np.arccos
            val_dict['sqrt'] = np.sqrt

            # Compute residue
            residue = np.array(ceval.eval_expr(val_dict), dtype=float)

            # Compute jacobian:
            J = np.array(Jeval.eval_expr(val_dict), dtype=float)

            # apply Newton-Rapshon step:
            vals_new = vals - np.linalg.lstsq(J, residue, rcond=None)[0].flatten()

            # Update dict
            val_dict = dict(zip([key for key in keys], vals_new.flatten()))
            iterations.append(val_dict)
            if np.abs(residue).sum() < 1e-12:
                #print(f'Converged in {cnt} steps')
                break
            elif cnt > 20:
                RuntimeWarning('Failed to converge')
                break

        return val_dict, residue, iterations


class AbstractConstraint(object):

    @property
    def equation(self):
        raise NotImplementedError()

    @property
    def elements(self):
        raise NotImplementedError()

    @property
    def symbols(self):
        raise NotImplementedError()

    def evaluate(self):
        # Get equation:
        evaluator = msym.EquationEvaluator(str(self.equation))
        # Get value_dict:
        val_dict = {'acos': np.arccos, 'sqrt': np.sqrt}
        for item in self.elements:
            for key, value in item.value_dict.items():
                val_dict[key] = value
        return evaluator.eval_expr(val_dict)

    def __str__(self):
        return f'Constraint {type(self)}, current residue: {self.evaluate()}'


class FixPoint(AbstractConstraint):

    def __init__(self, point: Point2D):
        self._point = point

    @property
    def equation(self):
        return [self._point.symbols['x'] - self._point.x,
                self._point.symbols['y'] - self._point.y]

    @property
    def elements(self):
        return [self._point]

    @property
    def symbols(self) -> list:
        return [self._point.symbols['x'], self._point.symbols['y']]


class Length(AbstractConstraint):

    def __init__(self, line: Line2D, length: float):
        self._line = line
        self._L = length

    @property
    def equation(self):
        """ Return length conserving equation
        """
        return [((self._line.p1.symbols['x'] - self._line.p2.symbols['x']) ** 2 +
                 (self._line.p1.symbols['y'] - self._line.p2.symbols['y']) ** 2) ** 0.5 - self._L]

    @property
    def elements(self):
        return [self._line]

    @property
    def symbols(self):
        return [item for _, item in self._line.symbols.items()]


class Coincide(AbstractConstraint):

    def __init__(self, p1: Point2D, p2: Point2D):
        self._p1 = p1
        self._p2 = p2

    @property
    def equation(self):
        return [self._p1.symbols['x'] - self._p2.symbols['x'],
                self._p1.symbols['y'] - self._p2.symbols['y'],
                ]

    @property
    def elements(self):
        return [self._p1, self._p2]

    @property
    def symbols(self):
        return [(item for _, item in p.symbols.items()) for p in [self._p1, self._p2]]


class Vertical(AbstractConstraint):

    def __init__(self, p1: Point2D, p2: Point2D):
        self._p1 = p1
        self._p2 = p2

    @property
    def equation(self):
        return [self._p1.symbols['x'] - self._p2.symbols['x']]

    @property
    def elements(self):
        return [self._p1, self._p2]

    @property
    def symbols(self):
        return [self._p1.symbols['x'], self._p2.symbols['x']]


class Horizontal(AbstractConstraint):

    def __init__(self, p1: Point2D, p2: Point2D):
        self._p1 = p1
        self._p2 = p2

    @property
    def equation(self):
        return [self._p1.symbols['y'] - self._p2.symbols['y']]

    @property
    def elements(self):
        return [self._p1, self._p2]

    @property
    def symbols(self):
        return [self._p1.symbols['y'], self._p1.symbols['y']]


class Angle(AbstractConstraint):

    def __init__(self, line: Line2D, angle: float):
        self._line = line
        self._angle = angle

    @property
    def equation(self):
        def vector_direction_ratio(x_1, y_1, x_2, y_2):
            return (y_1 * y_2 + x_1 * x_2) / ((x_1 ** 2 + y_1 ** 2) ** 0.5 * (x_2 ** 2 + y_2 ** 2) ** 0.5)

        ratio = vector_direction_ratio(self._line.p2.symbols['x'] - self._line.p1.symbols['x'],
                                       self._line.p2.symbols['y'] - self._line.p1.symbols['y'],
                                       1, 0)

        return [sp.acos(ratio) - self._angle]

    @property
    def elements(self):
        return [self._line]

    @property
    def symbols(self):
        return [item for _, item in self._line.symbols.items()]


class TangentLine(AbstractConstraint):

    def __init__(self, arc: Arc2D, angle_symbol, line: Line2D, connection_symbol: Point2D):
        self._arc = arc
        self._line = line

        self._angle_symbol = angle_symbol
        self._cp = connection_symbol

    @property
    def equation(self):

        def vector_direction_ratio(x_1, y_1, x_2, y_2):
            return (y_1 * y_2 + x_1 * x_2) / ((x_1 ** 2 + y_1 ** 2) ** 0.5 * (x_2 ** 2 + y_2 ** 2) ** 0.5)

        if self._line.p1 is self._cp:
            ratio = vector_direction_ratio(self._line.p2.symbols['x'] - self._cp.symbols['x'],
                                           self._line.p2.symbols['y'] - self._cp.symbols['y'],
                                           1, 0)
        elif self._line.p2 is self._cp:
            ratio = vector_direction_ratio(self._line.p1.symbols['x'] - self._cp.symbols['x'],
                                           self._line.p1.symbols['y'] - self._cp.symbols['y'],
                                           1, 0)

        return [
            # Ensure point intersects arc:
            self._arc.p_c.symbols['x'] + self._arc.r * sp.cos(self._angle_symbol) - self._cp.symbols['x'],
            self._arc.p_c.symbols['y'] + self._arc.r * sp.sin(self._angle_symbol) - self._cp.symbols['y'],
            # Ensure angle is tangential at intersection:
            sp.acos(ratio) - np.pi / 2 + self._angle_symbol]

    @property
    def elements(self):
        return [self._arc, self._line]

    @property
    def symbols(self):
        return [self._angle_symbol, self._cp.symbols['x'], self._cp.symbols['y'],
                self._arc.p_c.symbols['x'], self._arc.p_c.symbols['y']]


class TangentArc(AbstractConstraint):

    def __init__(self, arc1: Arc2D, angle_symbol1: sp.Symbol, arc2: Arc2D, angle_symbol2: sp.Symbol):
        self._arc1 = arc1
        self._arc2 = arc2

        self._angle_symbol1 = angle_symbol1
        self._angle_symbol2 = angle_symbol2

    @property
    def equation(self):
        return [
            # Ensure point intersects arc:
            self._arc1.p_c.symbols['x'] + self._arc1.r * sp.cos(self._angle_symbol1) -
            (self._arc2.p_c.symbols['x'] + self._arc2.r * sp.cos(self._angle_symbol2)),
            self._arc1.p_c.symbols['y'] + self._arc1.r * sp.sin(self._angle_symbol1) -
            (self._arc2.p_c.symbols['y'] + self._arc2.r * sp.sin(self._angle_symbol2)),
            # Ensure angle is tangential at intersection:
            self._angle_symbol1 - self._angle_symbol2]

    @property
    def elements(self):
        return [self._arc1, self._arc2]

    @property
    def symbols(self):
        return [self._angle_symbol1, self._angle_symbol2,
                self._arc1.p_c.symbols['x'], self._arc1.p_c.symbols['y'],
                self._arc2.p_c.symbols['x'], self._arc2.p_c.symbols['y'],
                ]