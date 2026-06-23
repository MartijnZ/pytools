import madernpytools.backbone as mbb
from madernpytools.models.toolset_model import BearerRing, steel, IToolset
from madernpytools.models.mechanics import IMaterial
import traitlets
import numpy as np


class EngToolsClassFactory(mbb.IClassFactory):

    @staticmethod
    def get(name):
        return eval(name)


# Convexity computer
class BearerMountingConditions(mbb.MadernObject, mbb.TraitsXMLSerializer):
    dT = traitlets.CFloat(80)
    diametrical_clearance = traitlets.CFloat(0.08)
    fit_pressure = traitlets.CFloat(80)

    def __init__(self, dT, fit_pressure, diametrical_clearance):
        """

        @param dT: Temperature difference between bore and shaft parts
        @param fit_pressure: Desired interference fit pressure
        @param diametrical_clearance: Required clearance for installation
        """
        super().__init__(dT=dT, fit_pressure=fit_pressure, diametrical_clearance=diametrical_clearance)


class BearerPressFitDesign(mbb.MadernObject):
    shaft_diameter = traitlets.Float()
    outer_diameter = traitlets.Float()
    material = IMaterial()
    mounting_conditions = BearerMountingConditions(dT=80, fit_pressure=18, diametrical_clearance=0.01)

    def __init__(self, outer_diameter:float, bearer_shaft_diameter: float, material: IMaterial,
                 mounting_conditions: BearerMountingConditions):
        """ Defines the bearer press fit design

        Details of these computations are given in D0092676 (bearer_design_computations)

        @param outer_diameter:  Bearer outer diameter
        @param bearer_shaft_diameter: (mm) Diameter of the shaft on which the bearer is mounted
        @param material: Bearer material
        @param mounting_conditions:  The mounting conditions
        """
        super().__init__(shaft_diameter=bearer_shaft_diameter, outer_diameter=outer_diameter,
                         material=material, mounting_conditions=mounting_conditions)

        self.observe(self.value_change, 'shaft_diameter', 'outer_diameter')

    @traitlets.observe('mounting_conditions', 'material')
    def cond_change(self, change):
        if isinstance(change.old, type(change.new)):
            change.old.unobserve(self.value_change)
        change.new.observe(self.value_change)

    def value_change(self, change):
        pass

    @property
    def interference(self):
        """ Returns the press-fit diametrical interference

        """
        p = self.mounting_conditions.fit_pressure  # Fit pressure (MPa)
        d_o = self.outer_diameter                 # Outer diameter (mm)
        d_in = self.shaft_diameter                 # Nominal Inner diameter (mm)
        E = self.material.E                 # MPa E-modulus

        d_od2 = (d_o / d_in) ** 2

        return p * (d_in / E) * ((d_od2 + 1) / (d_od2 - 1) + 1)

    @property
    def thermal_expansion(self):
        """ Returns the bearer thermal expansion during installation

        """
        d = self.shaft_diameter
        eps = self.material.eps
        dT = self.mounting_conditions.dT

        return d * eps * dT

    @property
    def br_bore_press_fit_diameter(self):
        """ The bearer bore, required to establish the desired interference pressure
        @return:
        """
        return self.shaft_diameter - self.interference

    @property
    def installation_clearance(self):
        """ Returns the clearance required for installation

        @return:
        """
        delta = self.interference
        dl_dT = self.thermal_expansion

        cl = dl_dT - delta

        return cl

    @property
    def meets_installation_requirements(self):
        return self.installation_clearance > self.mounting_conditions.diametrical_clearance


class ToolSetupParameters(mbb.MadernObject):
    s_init = traitlets.CFloat(1.0)
    s_sym = traitlets.CFloat(2.5)
    setup_supplement = traitlets.CFloat(0.0)
    gap = traitlets.CFloat(0.0)
    bearer_load = traitlets.CFloat(20e3)

    def __init__(self, s_init, s_sym, setup_supplement, gap, bearer_load):
        """

        @param s_init: (mm) Gap between body and bearer at initial tool setup (i.e. as manufactured)
        @param s_sym: (mm) Gap between body and bearer at which a symmetric slip-profile should be achieved
        @param setup_supplement: (mm) Supplement to achieve a desired gap
        @param gap: (mm) Desired gap
        @param bearer_load: (N) Tool loading
        """
        super().__init__(s_init=s_init, s_sym=s_sym, setup_supplement=setup_supplement,
                         bearer_load=bearer_load, gap=gap)


class IBearerGrindingDiameters(mbb.MadernObject):

    @property
    def symmetric_slip_diameter(self):
        raise NotImplementedError()

    @property
    def brmale_max_diameter(self):
        raise NotImplementedError()

    @property
    def brfemale_max_diameter(self):
        raise NotImplementedError()


class StaticGrindingDiameterCalculation(IBearerGrindingDiameters):
    tool_setup_parameters = ToolSetupParameters(s_init=1.0, s_sym=2.5, bearer_load=20e3, setup_supplement=5e-3, gap=1e-3)
    toolset = IToolset()

    def __init__(self, toolset: IToolset, tool_setup_parameters: ToolSetupParameters):
        """ This design object allows one to compute the outer bearer dimensions required to
        establish the desired tool-setup.

        Details of these computations are described in D0092676 (Bearer_design_computations)

        @param male_bearer: Male bearer object
        @param female_bearer: Female bearer object
        @param s_init: Initial gap between body and male bearer
        @param s_sym: Gap between body and male bearer for which symmetric slip should be achieved
        """
        super().__init__(toolset=toolset, tool_setup_parameters=tool_setup_parameters)


    @property
    def symmetric_slip_diameter(self):
        """ Returns the diameter at which 'zero' slip occurs when body<-> bearer distance is set to 's_sym'.

        @param d_m: (mm) Male image diameter
        @param d_f: (mm) Female image diameter
        @param s_sym:  (mm) body <-> bearer distance symmetric slip distribution range
        @param s_init:  (mm) body <-> bearer distance at assembly (initial setting)
        @param alpha: (deg) conical angle
        @param T: (mm) Required tolerance
        """
        d_m = self.toolset.upper_cylinder.diameter
        d_f = self.toolset.lower_cylinder.diameter
        s_sym = self.tool_setup_parameters.s_sym
        s_init = self.tool_setup_parameters.s_init
        alpha = self.toolset.upper_cylinder.bearer_ring.angle
        suppl = self.tool_setup_parameters.setup_supplement

        return 0.5 * (d_m + d_f) - (s_sym - s_init) * np.tan(alpha / 180 * np.pi) + suppl

    @property
    def brmale_max_diameter(self):
        """

        @param d_sym_slip:  (mm) Bearer ring diameter at 'symmetric-slip' condition
        @param w_br:  (mm) Male bearer width
        @param alpha: (deg) conical angle
        """
        w_br = self.toolset.upper_cylinder.bearer_ring.width
        alpha = self.toolset.upper_cylinder.bearer_ring.angle
        d_sym_slip = self.symmetric_slip_diameter

        return w_br * np.tan(alpha / 180 * np.pi) + d_sym_slip

    @property
    def brfemale_max_diameter(self):
        """


        @param d_sym_slip:  (mm) Bearer ring diameter at 'symmetric-slip' condition
        @param w_br:  (mm) Male bearer width
        @param s_sym:  (mm) body <-> bearer distance symmetric slip distribution range
        @param alpha: (deg) conical angle
        """
        w_br = self.toolset.upper_cylinder.bearer_ring.width
        alpha = self.toolset.upper_cylinder.bearer_ring.angle
        s_sym = self.tool_setup_parameters.s_sym
        d_sym_slip = self.symmetric_slip_diameter

        return (w_br + 2 * s_sym) * np.tan(alpha / 180 * np.pi) + d_sym_slip

# We keep a copy of StaticGrindingParameters under 'BearingGrindingDiameters' for backwards compatability:
class BearerGrindingDiameters(StaticGrindingDiameterCalculation):
    pass


class DynamicGrindingDiameterCalculation(IBearerGrindingDiameters):
    tool_setup_parameters = ToolSetupParameters(s_init=1.0, s_sym=2.5, bearer_load=20e3, setup_supplement=5e-3,
                                                gap=3e-3)
    toolset = IToolset()

    def __init__(self, toolset: IToolset, tool_setup_parameters: ToolSetupParameters):
        """ This design object allows one to compute the outer bearer dimensions required to
        establish the desired tool-setup.

        Details of these computations are described in D0092676 (Bearer_design_computations)

        @param male_bearer: Male bearer object
        @param female_bearer: Female bearer object
        @param s_init: Initial gap between body and male bearer
        @param s_sym: Gap between body and male bearer for which symmetric slip should be achieved
        """
        super().__init__(toolset=toolset, tool_setup_parameters=tool_setup_parameters)

    @property
    def setup_supplement(self):
        # Back-up current loads:
        q_cut = self.toolset.q_cut
        F_t = self.toolset.F_t

        # Set desired loads:
        self.toolset.q_cut = 0.0
        self.toolset.F_t = self.tool_setup_parameters.bearer_load

        # Get bearer deflection
        dx = self.toolset.deflections.bearer_deflection

        # Restore loads
        self.toolset.q_cut = q_cut
        self.toolset.F_t = F_t

        return dx + self.tool_setup_parameters.gap

    @property
    def symmetric_slip_diameter(self):
        """ Returns the diameter at which 'zero' slip occurs when body<-> bearer distance is set to 's_sym'.

        """
        d_m = self.toolset.upper_cylinder.diameter
        d_f = self.toolset.lower_cylinder.diameter
        s_sym = self.tool_setup_parameters.s_sym
        s_init = self.tool_setup_parameters.s_init
        alpha = self.toolset.upper_cylinder.bearer_ring.angle

        return 0.5 * (d_m + d_f) - (s_sym - s_init) * np.tan(alpha / 180 * np.pi) + self.setup_supplement

    @property
    def brmale_max_diameter(self):
        return StaticGrindingDiameterCalculation.brmale_max_diameter.__get__(self)

    @property
    def brfemale_max_diameter(self):
        return StaticGrindingDiameterCalculation.brfemale_max_diameter.__get__(self)


bearer_diameter_methods = [cls.__name__ for cls in IBearerGrindingDiameters.__subclasses__()]
