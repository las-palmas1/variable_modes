from .models import *
from gas_turbine_cycle.core.turbine_lib import CombustionChamber, Outlet, Inlet
from .compressor_characteristics.storage import Characteristics, From16To18Pi
from compressor.average_streamline.compressor import Compressor
from turbine.average_streamline.turbine import Turbine
from abc import ABCMeta, abstractmethod
from scipy.optimize import root
import numpy as np
import typing
import matplotlib.pyplot as plt
import pandas as pd
import enum


class SchemeSolvingOption(enum.Enum):
    POWER = 0
    CONST_TEMP = 1


class Scheme(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.x0: np.ndarray = None
        self.x_arr: typing.List[np.ndarray] = None
        self.modes_params: pd.DataFrame = None
        self.option_args = kwargs
        self.option: SchemeSolvingOption = None
        self.N_e = None
        self.C_e = None
        self.eta_e = None

    def get_option(self, option_args: dict):
        if self.get_power_option_condition(option_args):
            return SchemeSolvingOption.POWER
        elif self.get_const_temp_option_condition(option_args):
            return SchemeSolvingOption.CONST_TEMP
        else:
            raise Exception('Incorrect set of scheme solving options.')

    @abstractmethod
    def get_const_temp_option_condition(self, option_args: dict) -> bool:
        pass

    @abstractmethod
    def get_power_option_condition(self, option_args: dict) -> bool:
        pass

    @classmethod
    def set_inlet_params(cls, upstream_model: Model, downstream_model: Model):
        downstream_model.T_stag_in = upstream_model.T_stag_out
        downstream_model.p_stag_in = upstream_model.p_stag_out
        downstream_model.G_in = upstream_model.G_out
        downstream_model.G_fuel_in = upstream_model.G_fuel_out

    @abstractmethod
    def init_models_with_nominal_params(self):
        pass

    @abstractmethod
    def compute(self, *args):
        pass

    @abstractmethod
    def get_residuals(self, *args, **kwargs):
        pass

    @abstractmethod
    def _solve_with_power_option(self):
        pass

    @abstractmethod
    def _solve_with_const_temp_option(self):
        pass

    def solve(self):
        self.x_arr = []
        self.option = self.get_option(self.option_args)
        self.init_models_with_nominal_params()

        if self.option == SchemeSolvingOption.POWER:
            self._solve_with_power_option()
        elif self.option == SchemeSolvingOption.CONST_TEMP:
            self._solve_with_const_temp_option()
        self.compute_modes_params()

    @abstractmethod
    def compute_modes_params(self):
        pass

    def plot_characteristics(self, *args):
        pass

    @classmethod
    def plot_inlet_temp_plot(cls, T_stag_in_arr, value_arr, value_label, figsize=(7, 5), fname=None):
        plt.figure(figsize=figsize)
        plt.plot(T_stag_in_arr, value_arr, lw=1, color='orange')
        plt.plot(T_stag_in_arr, value_arr, ls='', color='orange', marker='o', markersize=8)
        plt.xlabel(r'$T_{вх}^*,\ C$', fontsize=12)
        plt.ylabel(value_label, fontsize=12)
        plt.xlim(min(T_stag_in_arr), max(T_stag_in_arr))
        plt.grid()
        plt.show()
        if fname:
            plt.savefig(fname)


class TwoShaftGeneratorVar1(Scheme):
    def __init__(
            self, inlet: Inlet, compressor: Compressor, comp_turbine: Turbine, power_turbine: Turbine,
            comb_chamber: CombustionChamber, outlet: Outlet, p_stag_in,
            T_a, p_a, g_cool_sum, g_cool_st1, g_cool_st2, eta_r, pi_c_stag_rel_init, n_norm_rel_init,
            pi_t1_stag_init, pi_t2_stag_init, pi_t3_stag_init,
            comp_char: Characteristics=From16To18Pi(), outlet_diff_coef=2, precision=0.0001, **kwargs
    ):
        Scheme.__init__(self)
        self.inlet = inlet
        self.compressor = compressor
        self.comp_turbine = comp_turbine
        self.power_turbine = power_turbine
        self.comb_chamber = comb_chamber
        self.p_stag_in = p_stag_in
        self.T_a = T_a
        self.p_a = p_a
        self.g_cool_sum = g_cool_sum
        self.g_cool_st1 = g_cool_st1
        self.g_cool_st2 = g_cool_st2
        self.eta_r = eta_r
        self.outlet = outlet
        self.outlet_diff_coef = outlet_diff_coef
        self.comp_char = comp_char
        self.pi_c_stag_rel_init = pi_c_stag_rel_init
        self.n_norm_rel_init = n_norm_rel_init
        self.pi_t1_stag_init = pi_t1_stag_init
        self.pi_t2_stag_init = pi_t2_stag_init
        self.pi_t3_stag_init = pi_t3_stag_init
        self.precision = precision
        self.option_args = kwargs

    def get_const_temp_option_condition(self, option_args: dict) -> bool:
        return 'T_g_stag_arr' in option_args and 'g_fuel_init' in option_args and 'T_stag_in' in option_args

    def get_power_option_condition(self, option_args: dict) -> bool:
        return 'g_fuel' in option_args and 'T_stag_in_arr' in option_args

    def init_models_with_nominal_params(self):
        self.comp_model = CompressorModel(
            characteristics=self.comp_char,
            T_stag_in=None,
            p_stag_in=self.p_stag_in,
            pi_c_stag_rel=None,
            n_norm_rel=None,
            T_stag_in_nom=self.inlet.T_stag_in,
            p_stag_in_nom=self.inlet.p_stag_in,
            G_in_nom=self.compressor.G,
            eta_c_stag_nom=self.compressor.eta_c_stag,
            n_nom=self.compressor.n,
            pi_c_stag_nom=self.compressor.pi_c_stag,
            p_a=self.p_a,
            T_a=self.T_a,
            G_fuel_in=0,
            work_fluid=type(self.compressor.work_fluid)(),
            sigma_in=self.inlet.sigma,
            precision=self.precision
        )
        self.sink_model = SinkModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            G_c1_in=None,
            g_cool=self.g_cool_sum
        )
        self.comb_chamber_model = CombustionChamberModel(
            g_fuel=None,
            G_c1_in=None,
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            eta_comb=self.comb_chamber.eta_burn,
            sigma_comb=self.comb_chamber.sigma_comb,
            T_fuel=self.comb_chamber.T_fuel,
            fuel=self.comb_chamber.fuel,
            delta_p_fuel=self.comb_chamber.delta_p_fuel,
            work_fluid_in=type(self.comb_chamber.work_fluid_in)(),
            work_fluid_out=type(self.comb_chamber.work_fluid_out)(),
            precision=self.precision
        )
        self.comp_turb_st1_model = TurbineModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            pi_t_stag=None,
            T_stag_in_nom=self.comp_turbine[0].T0_stag,
            p_stag_in_nom=self.comp_turbine[0].p0_stag,
            G_in_nom=self.comp_turbine[0].G_stage_in,
            pi_t_stag_nom=self.comp_turbine[0].pi_stag,
            eta_t_stag_nom=self.comp_turbine[0].eta_t_stag,
            eta_m=self.comp_turbine.eta_m,
            work_fluid=type(self.comp_turbine.work_fluid)()
        )
        self.source_st1_model = SourceModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            g_cool=self.g_cool_st1,
            G_c1_in=None,
            T_cool=self.comp_turbine[0].T_cool,
            cool_fluid=type(self.comp_turbine[0].cool_fluid)(),
            work_fluid=type(self.comp_turbine[0].work_fluid)(),
            precision=self.precision
        )
        self.comp_turb_st2_model = TurbineModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            pi_t_stag=None,
            T_stag_in_nom=self.comp_turbine[1].T0_stag,
            p_stag_in_nom=self.comp_turbine[1].p0_stag,
            G_in_nom=self.comp_turbine[1].G_stage_in,
            pi_t_stag_nom=self.comp_turbine[1].pi_stag,
            eta_t_stag_nom=self.comp_turbine[1].eta_t_stag,
            eta_m=self.comp_turbine.eta_m,
            work_fluid=type(self.comp_turbine.work_fluid)()
        )
        self.source_st2_model = SourceModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            g_cool=self.g_cool_st2,
            G_c1_in=None,
            T_cool=self.comp_turbine[1].T_cool,
            cool_fluid=type(self.comp_turbine[1].cool_fluid)(),
            work_fluid=type(self.comp_turbine[1].work_fluid)(),
            precision=self.precision
        )
        self.power_turb_model = OutletTurbineModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            pi_t_stag=None,
            T_stag_in_nom=self.power_turbine.T_g_stag,
            p_stag_in_nom=self.power_turbine.p_g_stag,
            G_in_nom=self.power_turbine.G_turbine,
            pi_t_stag_nom=self.power_turbine.pi_t_stag,
            eta_t_stag_nom=self.power_turbine.eta_t_stag,
            F_out=self.power_turbine.geom.last.A2,
            eta_m=self.power_turbine.eta_m,
            work_fluid=type(self.power_turbine.work_fluid)()
        )
        self.outlet_model = OutletModel(
            T_stag_in=None,
            p_stag_in=None,
            G_in=None,
            G_fuel_in=None,
            c_in=None,
            F_in=self.power_turbine.geom.last.A2,
            F_out=self.power_turbine.geom.last.A2 * self.outlet_diff_coef,
            sigma=self.outlet.sigma,
            work_fluid=type(self.outlet.work_fluid)()
        )

    def compute(self, pi_c_stag_rel, n_norm_rel, pi_t1_stag, pi_t2_stag, pi_t3_stag, g_fuel, T_stag_in):
        self.comp_model.T_stag_in = T_stag_in
        self.comp_model.pi_c_stag_rel = pi_c_stag_rel
        self.comp_model.n_norm_rel = n_norm_rel
        self.comp_model.compute()

        self.set_inlet_params(self.comp_model, self.sink_model)
        self.sink_model.G_c1_in = self.comp_model.G_in
        self.sink_model.compute()

        self.set_inlet_params(self.sink_model, self.comb_chamber_model)
        self.comb_chamber_model.g_fuel = g_fuel
        self.comb_chamber_model.G_c1_in = self.comp_model.G_in
        self.comb_chamber_model.compute()

        self.set_inlet_params(self.comb_chamber_model, self.comp_turb_st1_model)
        self.comp_turb_st1_model.pi_t_stag = pi_t1_stag
        self.comp_turb_st1_model.compute()

        self.set_inlet_params(self.comp_turb_st1_model, self.source_st1_model)
        self.source_st1_model.G_c1_in = self.comp_model.G_in
        self.source_st1_model.compute()

        self.set_inlet_params(self.source_st1_model, self.comp_turb_st2_model)
        self.comp_turb_st2_model.pi_t_stag = pi_t2_stag
        self.comp_turb_st2_model.compute()

        self.set_inlet_params(self.comp_turb_st2_model, self.source_st2_model)
        self.source_st2_model.G_c1_in = self.comp_model.G_in
        self.source_st2_model.compute()

        self.set_inlet_params(self.source_st2_model, self.power_turb_model)
        self.power_turb_model.pi_t_stag = pi_t3_stag
        self.power_turb_model.compute()

        self.set_inlet_params(self.power_turb_model, self.outlet_model)
        self.outlet_model.c_in = self.power_turb_model.c_out
        self.outlet_model.compute()

        self.N_e = -self.power_turb_model.N * self.eta_r
        self.C_e = self.comb_chamber_model.G_fuel * 3600 / self.N_e
        self.eta_e = 3600 / (self.C_e * self.comb_chamber_model.work_fluid_out.Q_n)

    def get_residuals(self, pi_c_stag_rel, n_norm_rel, pi_t1_stag, pi_t2_stag, pi_t3_stag, g_fuel, T_stag_in, **kwargs):
        print('Residual computing.')
        print('pi_c_stag_rel = %.3f' % pi_c_stag_rel)
        print('n_norm_rel = %.3f' % n_norm_rel)
        print('pi_t1_stag = %.2f' % pi_t1_stag)
        print('pi_t2_stag = %.2f' % pi_t2_stag)
        print('pi_t3_stag = %.2f' % pi_t3_stag)
        print('g_fuel = %.4f' % g_fuel)
        print('T_stag_in = %.2f\n' % T_stag_in)
        self.compute(pi_c_stag_rel, n_norm_rel, pi_t1_stag, pi_t2_stag, pi_t3_stag, g_fuel, T_stag_in)

        G_t1_res = self.comp_turb_st1_model.G_in - self.comp_turb_st1_model.G_in_char
        G_t2_res = self.comp_turb_st2_model.G_in - self.comp_turb_st2_model.G_in_char
        G_t3_res = self.power_turb_model.G_in - self.power_turb_model.G_in_char
        L_res = self.comp_turb_st1_model.N + self.comp_turb_st2_model.N + self.comp_model.N
        p_out_res = self.outlet_model.p_out - self.p_a

        print('G_t1_res = %.3f' % G_t1_res)
        print('G_t2_res = %.3f' % G_t2_res)
        print('G_t3_res = %.3f' % G_t3_res)
        print('L_res = %.3f' % L_res)
        print('p_out_res = %.3f\n' % p_out_res)

        if 'T_g_stag' in kwargs:
            T_res = kwargs['T_g_stag'] - self.comb_chamber.T_stag_out
            return G_t1_res, G_t2_res, G_t3_res, L_res, p_out_res, T_res
        else:
            return G_t1_res, G_t2_res, G_t3_res, L_res, p_out_res

    def compute_modes_params(self):
        if len(self.x_arr) == 0:
            raise Exception('Scheme has not been computed.')

        self.modes_params = pd.DataFrame.from_dict({
            'pi_c_stag_rel': [],
            'n_norm_rel': [],
            'pi_t1_stag': [],
            'pi_t2_stag': [],
            'pi_t3_stag': [],
            'T_stag_in': [],
            'T_g_stag': [],
            'g_fuel': [],
            'G_fuel': [],
            'N_e': [],
            'C_e': [],
            'eta_e': [],
            'G_in_norm_rel': [],
            'eta_c_stag_rel': [],
            'G_c': [],
            'eta_c_stag': [],
            'pi_c_stag': [],
            'n': [],
            'G_out': [],
            'T_out': []
        })

        if self.option == SchemeSolvingOption.POWER:
            for x, T_stag_in in zip(self.x_arr, self.option_args['T_stag_in_arr']):
                self.compute(*x, g_fuel=self.option_args['g_fuel'], T_stag_in=T_stag_in)
                self.modes_params = self.modes_params.append(pd.DataFrame.from_dict({
                    'pi_c_stag_rel': [self.comp_model.pi_c_stag_rel],
                    'n_norm_rel': [self.comp_model.n_norm_rel],
                    'pi_t1_stag': [self.comp_turb_st1_model.pi_t_stag],
                    'pi_t2_stag': [self.comp_turb_st2_model.pi_t_stag],
                    'pi_t3_stag': [self.power_turbine.pi_t_stag],
                    'T_stag_in': [T_stag_in],
                    'T_g_stag': [self.comb_chamber_model.T_stag_out],
                    'g_fuel': [self.comb_chamber_model.g_fuel],
                    'G_fuel': [self.comb_chamber_model.G_fuel],
                    'N_e': [self.N_e],
                    'C_e': [self.C_e],
                    'eta_e': [self.eta_e],
                    'G_in_norm_rel': [self.comp_model.G_in_norm_rel],
                    'eta_c_stag_rel': [self.comp_model.eta_c_stag_rel],
                    'G_c': [self.comp_model.G_in],
                    'eta_c_stag': [self.comp_model.eta_c_stag],
                    'pi_c_stag': [self.comp_model.pi_c_stag],
                    'n': [self.comp_model.n],
                    'G_out': [self.outlet_model.G_out],
                    'T_out': [self.outlet_model.T_stag_out]
                }), ignore_index=True)
        elif self.option == SchemeSolvingOption.CONST_TEMP:
            for x, T_g_stag in zip(self.x_arr, self.option_args['T_g_stag_arr']):
                self.compute(*x, T_stag_in=self.option_args['T_stag_in'])
                self.modes_params = self.modes_params.append(pd.DataFrame.from_dict({
                    'pi_c_stag_rel': [self.comp_model.pi_c_stag_rel],
                    'n_norm_rel': [self.comp_model.n_norm_rel],
                    'pi_t1_stag': [self.comp_turb_st1_model.pi_t_stag],
                    'pi_t2_stag': [self.comp_turb_st2_model.pi_t_stag],
                    'pi_t3_stag': [self.power_turbine.pi_t_stag],
                    'T_stag_in': [self.option_args['T_stag_in']],
                    'T_g_stag': [self.comb_chamber_model.T_stag_out],
                    'g_fuel': [self.comb_chamber_model.g_fuel],
                    'G_fuel': [self.comb_chamber_model.G_fuel],
                    'N_e': [self.N_e],
                    'C_e': [self.C_e],
                    'eta_e': [self.eta_e],
                    'G_in_norm_rel': [self.comp_model.G_in_norm_rel],
                    'eta_c_stag_rel': [self.comp_model.eta_c_stag_rel],
                    'G_c': [self.comp_model.G_in],
                    'eta_c_stag': [self.comp_model.eta_c_stag],
                    'pi_c_stag': [self.comp_model.pi_c_stag],
                    'n': [self.comp_model.n],
                    'G_out': [self.outlet_model.G_out],
                    'T_out': [self.outlet_model.T_stag_out]
                }), ignore_index=True)

    def _solve_with_power_option(self):
        self.x0 = np.array([
            self.pi_c_stag_rel_init, self.n_norm_rel_init, self.pi_t1_stag_init,
            self.pi_t2_stag_init, self.pi_t3_stag_init
        ])
        for n, T_stag_in in enumerate(self.option_args['T_stag_in_arr']):
            if n == 0:
                sol = root(
                    fun=lambda x: self.get_residuals(x[0], x[1], x[2], x[3], x[4],
                                                     self.option_args['g_fuel'], T_stag_in),
                    x0=self.x0, method='hybr'
                )
                self.x_arr.append(sol.x)
            else:
                sol = root(
                    fun=lambda x: self.get_residuals(x[0], x[1], x[2], x[3], x[4],
                                                     self.option_args['g_fuel'], T_stag_in),
                    x0=self.x_arr[n - 1], method='hybr'
                )
                self.x_arr.append(sol.x)

    def _solve_with_const_temp_option(self):
        self.x0 = np.array([
            self.pi_c_stag_rel_init, self.n_norm_rel_init, self.pi_t1_stag_init,
            self.pi_t2_stag_init, self.pi_t3_stag_init, self.option_args['g_fuel_init']
        ])
        for n, T_g_stag in enumerate(self.option_args['T_g_stag_arr']):
            if n == 0:
                sol = root(
                    fun=lambda x: self.get_residuals(x[0], x[1], x[2], x[3], x[4], x[5],
                                                     self.option_args['T_stag_in'], T_g_stag=T_g_stag),
                    x0=self.x0, method='hybr'
                )
                self.x_arr.append(sol.x)
            else:
                sol = root(
                    fun=lambda x: self.get_residuals(x[0], x[1], x[2], x[3], x[4], x[5],
                                                     self.option_args['T_stag_in'], T_g_stag=T_g_stag),
                    x0=self.x_arr[n - 1], method='hybr'
                )
                self.x_arr.append(sol.x)


