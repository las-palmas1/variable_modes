from gas_turbine_cycle.gases import IdealGas, Air, NaturalGasCombustionProducts
from gas_turbine_cycle.fuels import Fuel, NaturalGas
from gas_turbine_cycle.tools.functions import get_mixture_temp
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
from .compressor_characteristics.tools import Characteristics
import numpy as np
from scipy.optimize import newton, fsolve


class Model:
    def __init__(self):
        self.p_stag_in = None
        self.T_stag_in = None
        self.G_in = None
        self.G_fuel_in = None
        self.p_stag_out = None
        self.T_stag_out = None
        self.G_out = None
        self.G_fuel_out = None

    def compute(self):
        pass


class ModelStaticPar(Model):
    def __init__(self):
        Model.__init__(self)
        self.T_in = None
        self.p_in = None
        self.T_out = None
        self.p_out = None
        self.rho_in = None
        self.rho_out = None
        self.c_in = None
        self.c_out = None
        self.lam_in = None
        self.lam_out = None

    def compute(self):
        pass


class ModelMultipleWorkFluid(Model):
    def __init__(self):
        Model.__init__(self)
        self.work_fluid_in: IdealGas = None
        self.work_fluid_out: IdealGas = None

    def compute(self):
        pass


class ModelSingleWorkFluid(Model):
    def __init__(self):
        Model.__init__(self)
        self.work_fluid = None

    def compute(self):
        pass


class CompressorModel(ModelSingleWorkFluid):
    def __init__(
            self, characteristics: Characteristics, T_stag_in, p_stag_in, pi_c_stag_rel, n_norm_rel,
            T_stag_in_nom, p_stag_in_nom, G_in_nom, eta_c_stag_nom, n_nom, pi_c_stag_nom,
            G_fuel_in=0, p_a=1e5, T_a=288, sigma_in=0.99, work_fluid: IdealGas=Air(), precision=0.0001,
    ):
        ModelSingleWorkFluid.__init__(self)
        self.characteristics = characteristics
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.pi_c_stag_rel = pi_c_stag_rel
        self.n_norm_rel = n_norm_rel
        self.T_stag_in_nom = T_stag_in_nom
        self.p_stag_in_nom = p_stag_in_nom
        self.G_in_nom = G_in_nom
        self.eta_c_stag_nom = eta_c_stag_nom
        self.n_nom = n_nom
        self.pi_c_stag_nom = pi_c_stag_nom
        self.G_fuel_in = G_fuel_in
        self.p_a = p_a
        self.T_a = T_a
        self.sigma_in = sigma_in
        self.work_fluid = work_fluid
        self.precision = precision

        self.n_norm_nom = None
        self.G_in_norm_nom = None
        self.G_in_norm_rel = None
        self.G_in_norm = None
        self.n_norm = None
        self.n = None
        self.eta_c_stag_rel = None
        self.pi_c_stag = None
        self.eta_c_stag = None
        self.ad_proc_res = None
        self.H_stag = None
        self.c_p_av_old = None
        self.k_av_old = None
        self.k_av = None
        self.k_res = None
        self.L = None
        self.T_stag_out_ad = None
        self.N = None

    def compute(self):
        self.characteristics.compute()
        self.G_fuel_out = self.G_fuel_in
        self.n_norm_nom = self.n_nom * np.sqrt(self.T_a / self.T_stag_in_nom)
        self.G_in_norm_nom = self.G_in_nom * self.p_a / self.p_stag_in_nom * np.sqrt(self.T_stag_in_nom / self.T_a)
        self.G_in_norm_rel = self.characteristics.get_G_rel(self.n_norm_rel, self.pi_c_stag_rel)
        self.G_in_norm = self.G_in_norm_nom * self.G_in_norm_rel
        self.G_in = self.G_in_norm * self.p_stag_in / self.p_a * np.sqrt(self.T_a / self.T_stag_in)
        self.n_norm = self.n_norm_rel * self.n_norm_nom
        self.n = self.n_norm * np.sqrt(self.T_stag_in / self.T_a)
        self.eta_c_stag_rel = self.characteristics.get_eta_c_stag_rel(self.n_norm_rel, self.G_in_norm_rel)
        self.pi_c_stag = self.pi_c_stag_rel * self.pi_c_stag_nom
        self.eta_c_stag = self.eta_c_stag_rel * self.eta_c_stag_nom
        self.p_stag_out = self.p_stag_in * self.pi_c_stag * self.sigma_in
        self.ad_proc_res = self.work_fluid.get_ad_temp(self.T_stag_in, self.p_stag_in * self.sigma_in,
                                                       self.p_stag_out, self.precision)
        self.T_stag_out_ad = self.ad_proc_res[0]
        self.H_stag = self.ad_proc_res[1]
        self.L = self.H_stag / self.eta_c_stag
        self.T_stag_out = self.work_fluid.get_temp(self.T_stag_in, self.L, self.precision)
        self.G_out = self.G_in
        self.N = self.L * self.G_in


class SinkModel(Model):
    def __init__(self, T_stag_in, p_stag_in, G_in, G_fuel_in, G_c1_in, g_cool):
        Model.__init__(self)
        self.G_c1_in = G_c1_in
        self.g_cool = g_cool
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.G_in = G_in
        self.G_fuel_in = G_fuel_in

    def compute(self):
        self.T_stag_out = self.T_stag_in
        self.p_stag_out = self.p_stag_in
        self.G_fuel_out = self.G_fuel_in
        self.G_out = self.G_in - self.G_c1_in * self.g_cool


class CombustionChamberModel(ModelMultipleWorkFluid):
    def __init__(
            self, g_fuel, G_c1_in, T_stag_in, p_stag_in, G_in, G_fuel_in, eta_comb, sigma_comb,
            T_fuel=300, fuel: Fuel=NaturalGas(),
            delta_p_fuel=5e5, work_fluid_in: IdealGas=Air(),
            work_fluid_out: IdealGas=NaturalGasCombustionProducts(),
            precision=0.0001
    ):
        ModelMultipleWorkFluid.__init__(self)
        self.g_fuel = g_fuel
        self.G_c1_in = G_c1_in
        self.G_fuel_in = G_fuel_in
        self.G_in = G_in
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.eta_comb = eta_comb
        self.sigma_comb = sigma_comb
        self.T_fuel = T_fuel
        self.fuel = fuel
        self.delta_p_fuel = delta_p_fuel
        self.work_fluid_in = work_fluid_in
        self.work_fluid_out = work_fluid_out
        self.precision = precision

        self.G_fuel = None
        self.fuel_content_out = None
        self.fuel_content_in = None
        self.alpha_in = None
        self.alpha_out = None
        self.i_stag_in = None
        self.i_stag_out = None
        self.p_fuel = None
        self.i_fuel = None

    def compute(self):
        self.G_fuel = self.g_fuel * self.G_c1_in
        self.G_fuel_out = self.G_fuel_in + self.G_fuel
        self.G_out = self.G_in + self.G_fuel
        self.p_stag_out = self.p_stag_in * self.sigma_comb
        self.fuel_content_out = self.G_fuel_out / (self.G_out - self.G_fuel_out)
        self.fuel_content_in = self.G_fuel_in / (self.G_in - self.G_fuel_in)
        if self.fuel_content_in != 0:
            self.alpha_in = 1 / (self.work_fluid_in.l0 * self.fuel_content_in)
        else:
            self.alpha_in = 1e5
        self.alpha_out = 1 / (self.work_fluid_out.l0 * self.fuel_content_out)
        self.i_stag_in = self.work_fluid_in.get_specific_enthalpy(self.T_stag_in, alpha=self.alpha_in)
        self.p_fuel = self.p_stag_in + self.delta_p_fuel
        self.i_fuel = self.fuel.get_specific_enthalpy(self.T_stag_in, p=self.p_fuel)
        self.i_stag_out = (
                self.G_in * self.i_stag_in + self.G_fuel * (self.work_fluid_out.Q_n * self.eta_comb + self.i_fuel)
        ) / self.G_out
        self.T_stag_out = self.work_fluid_out.get_temp(self.work_fluid_out.T0, self.i_stag_out, self.precision,
                                                       alpha=self.alpha_out)


class TurbineModel(ModelSingleWorkFluid):
    def __init__(
            self, T_stag_in, p_stag_in, G_in, G_fuel_in, pi_t_stag, T_stag_in_nom, p_stag_in_nom,
            G_in_nom, pi_t_stag_nom, eta_t_stag_nom, eta_m=0.9, work_fluid: IdealGas=NaturalGasCombustionProducts(),
            precision=0.001,
    ):
        ModelSingleWorkFluid.__init__(self)
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.G_in = G_in
        self.G_fuel_in = G_fuel_in
        self.pi_t_stag = pi_t_stag
        self.T_stag_in_nom = T_stag_in_nom
        self.p_stag_in_nom = p_stag_in_nom
        self.G_in_nom = G_in_nom
        self.pi_t_stag_nom = pi_t_stag_nom
        self.eta_t_stag_nom = eta_t_stag_nom
        self.eta_m = eta_m
        self.work_fluid = work_fluid
        self.precision = precision

        self.pi_t_stag_rel = None
        self.eta_t_stag_rel = None
        self.eta_t_stag = None
        self.fuel_content = None
        self.alpha = None
        self.ad_proc_res = None
        self.T_stag_out_ad = None
        self.H_stag = None
        self.L = None
        self.G_in_char = None
        self.N = None

    def compute(self):
        self.pi_t_stag_rel = self.pi_t_stag / self.pi_t_stag_nom
        self.eta_t_stag_rel = 1 - (1 - self.pi_t_stag_rel)**2 - 0.6 * (1 - self.pi_t_stag_rel)**3
        self.eta_t_stag = self.eta_t_stag_rel * self.eta_t_stag_nom
        self.p_stag_out = self.p_stag_in / self.pi_t_stag
        self.G_out = self.G_in
        self.G_fuel_out = self.G_fuel_in
        self.fuel_content = self.G_fuel_in / (self.G_in - self.G_fuel_in)
        self.alpha = 1 / (self.work_fluid.l0 * self.fuel_content)
        self.ad_proc_res = self.work_fluid.get_ad_temp(self.T_stag_in, self.p_stag_in, self.p_stag_out,
                                                       self.precision, alpha=self.alpha)
        self.T_stag_out_ad = self.ad_proc_res[0]
        self.H_stag = self.ad_proc_res[1]
        self.L = self.H_stag * self.eta_t_stag
        self.T_stag_out = self.work_fluid.get_temp(self.T_stag_in, self.L, self.precision, alpha=self.alpha)
        self.G_in_char = (
                self.G_in_nom * np.sqrt(self.T_stag_in_nom / self.T_stag_in) *
                (self.p_stag_in / self.p_stag_in_nom) *
                np.sqrt((1 - self.pi_t_stag**(-2)) / (1 - self.pi_t_stag_nom**(-2)))
        )
        self.N = self.L * self.G_in * self.eta_m


class OutletTurbineModel(TurbineModel):
    def __init__(
            self, T_stag_in, p_stag_in, G_in, G_fuel_in, pi_t_stag, T_stag_in_nom, p_stag_in_nom,
            G_in_nom, pi_t_stag_nom, eta_t_stag_nom, F_out,
            eta_m=0.9, work_fluid: IdealGas=NaturalGasCombustionProducts(),
            precision=0.001
    ):
        TurbineModel.__init__(self, T_stag_in, p_stag_in, G_in, G_fuel_in, pi_t_stag, T_stag_in_nom, p_stag_in_nom,
                              G_in_nom, pi_t_stag_nom, eta_t_stag_nom, eta_m, work_fluid, precision)
        self.F_out = F_out
        self.c_p_out = None
        self.k_out = None
        self.q_out = None
        self.lam_out = None
        self.a_cr_out = None
        self.c_out = None

    def compute(self):
        super(OutletTurbineModel, self).compute()
        self.c_p_out = self.work_fluid.c_p_real_func(self.T_stag_out, alpha=self.alpha)
        self.k_out = self.work_fluid.k_func(self.c_p_out)
        self.q_out = self.G_out * np.sqrt(self.work_fluid.R * self.T_stag_out) / (gd.m(self.k_out) * self.F_out *
                                                                                  self.p_stag_out)
        self.lam_out = gd.lam(self.k_out, q=self.q_out)
        self.a_cr_out = gd.a_cr(self.T_stag_out, self.k_out, self.work_fluid.R)
        self.c_out = self.lam_out * self.a_cr_out


class SourceModel(ModelSingleWorkFluid):
    def __init__(
            self, T_stag_in, p_stag_in, G_in, G_fuel_in, g_cool, G_c1_in, T_cool, cool_fluid: IdealGas=Air(),
            work_fluid: IdealGas=NaturalGasCombustionProducts(), precision=0.0001
    ):
        ModelSingleWorkFluid.__init__(self)
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.G_in = G_in
        self.G_fuel_in = G_fuel_in
        self.g_cool = g_cool
        self.G_c1_in = G_c1_in
        self.T_cool = T_cool
        self.cool_fluid = cool_fluid
        self.work_fluid = work_fluid
        self.precision = precision

        self.G_cool = None
        self.fuel_content_in = None
        self.fuel_content_out = None
        self.alpha_in = None
        self.alpha_out = None

    def compute(self):
        self.G_fuel_out = self.G_fuel_in
        self.G_cool = self.g_cool * self.G_c1_in
        self.G_out = self.G_in + self.G_cool
        self.fuel_content_in = self.G_fuel_in / (self.G_in - self.G_fuel_in)
        self.fuel_content_out = self.G_fuel_out / (self.G_out - self.G_fuel_out)
        self.alpha_in = 1 / (self.work_fluid.l0 * self.fuel_content_in)
        self.alpha_out = 1 / (self.work_fluid.l0 * self.fuel_content_out)
        self.p_stag_out = self.p_stag_in
        self.work_fluid.alpha = self.alpha_in
        self.T_stag_out = get_mixture_temp(
            self.work_fluid, self.cool_fluid, self.T_stag_in, self.T_cool, self.G_in, self.G_cool, self.alpha_out,
            self.precision
        )[0]


class OutletModel(ModelSingleWorkFluid, ModelStaticPar):
    def __init__(
            self, T_stag_in, p_stag_in, G_in, G_fuel_in, c_in, F_in, F_out, sigma,
            work_fluid: IdealGas=NaturalGasCombustionProducts()
    ):
        ModelSingleWorkFluid.__init__(self)
        ModelStaticPar.__init__(self)
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.G_in = G_in
        self.G_fuel_in = G_fuel_in
        self.c_in = c_in
        self.F_in = F_in
        self.F_out = F_out
        self.sigma = sigma
        self.work_fluid = work_fluid
        self.fuel_content = None
        self.alpha = None
        self.c_p = None
        self.k = None
        self.static_out = None

    @classmethod
    def get_static(cls, c, T_stag, p_stag, k, R):
        a_cr = gd.a_cr(T_stag, k, R)
        lam = c / a_cr
        tau = gd.tau_lam(lam, k)
        pi = gd.pi_lam(lam, k)
        T = T_stag * tau
        p = p_stag * pi
        rho = p / (R * T)
        return T, p, rho, lam

    def compute(self):
        self.fuel_content = self.G_fuel_in / (self.G_in - self.G_fuel_in)
        self.alpha = 1 / (self.work_fluid.l0 * self.fuel_content)
        self.G_fuel_out = self.G_fuel_in
        self.G_out = self.G_in
        self.c_p = self.work_fluid.c_p_real_func(self.T_stag_in, alpha=self.alpha)
        self.k = self.work_fluid.k_func(self.c_p)
        self.T_stag_out = self.T_stag_in
        self.p_stag_out = self.p_stag_in * self.sigma
        self.T_in, self.p_in, self.rho_in, self.lam_in = self.get_static(
            self.c_in, self.T_stag_in, self.p_stag_in, self.k, self.work_fluid.R
        )
        self.c_out = newton(
            lambda c: self.F_out * c * self.get_static(c, self.T_stag_out, self.p_stag_out,
                                                        self.k, self.work_fluid.R)[2] - self.G_out,
            x0=self.c_in * 0.9
        )
        self.static_out = self.get_static(self.c_out, self.T_stag_out, self.p_stag_out, self.k, self.work_fluid.R)
        self.T_out = self.static_out[0]
        self.p_out = self.static_out[1]
        self.rho_out = self.static_out[2]
        self.lam_out = self.static_out[3]





