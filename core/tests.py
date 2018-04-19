from .models import CombustionChamberModel, OutletTurbineModel, CompressorModel
from .compressor_characteristics.storage import From16To18Pi
from compressor.average_streamline.compressor import Compressor
from compressor.average_streamline.dist_tools import QuadraticBezier
import unittest
import numpy as np
from turbine.average_streamline.turbine import Turbine, TurbineType
from gas_turbine_cycle.gases import NaturalGasCombustionProducts, Air
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd


class ModelsTests(unittest.TestCase):
    def setUp(self):
        self.comb_chamber = CombustionChamberModel(
            g_fuel=0.02,
            G_c1_in=30,
            T_stag_in=800,
            p_stag_in=13e5,
            G_in=29,
            G_fuel_in=0,
            eta_comb=0.99,
            sigma_comb=0.99,
        )
        self.turbine = Turbine(
            turbine_type=TurbineType.WORK,
            stage_number=2,
            T_g_stag=1400,
            p_g_stag=15e5,
            G_turbine=34,
            G_fuel=0.9,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.15,
            n=10e3,
            T_t_stag_cycle=1150,
            eta_t_stag_cycle=0.9,
            alpha11=np.radians(14),
            precision=0.0001,
            gamma_in=np.radians(0),
            gamma_out=np.radians(10),
            H01_init=200e3,
            c21_init=250
        )
        self.turbine.compute_geometry()
        self.turbine.compute_stages_gas_dynamics()
        self.turbine.compute_integrate_turbine_parameters()
        self.turb_model = OutletTurbineModel(
            T_stag_in=self.turbine.T_g_stag,
            p_stag_in=self.turbine.p_g_stag,
            G_in=self.turbine.G_turbine,
            G_fuel_in=self.turbine.G_fuel,
            pi_t_stag=self.turbine.pi_t_stag,
            T_stag_in_nom=self.turbine.T_g_stag,
            p_stag_in_nom=self.turbine.p_g_stag,
            G_in_nom=self.turbine.G_turbine,
            pi_t_stag_nom=self.turbine.pi_t_stag,
            eta_t_stag_nom=self.turbine.eta_t_stag,
            F_out=self.turbine.geom.last.A2,
            eta_m=self.turbine.eta_m,
            precision=self.turbine.precision
        )
        comp_stage_num = 10
        H_t_rel_dist = QuadraticBezier(0.3, 0.25, stage_num=comp_stage_num,
                                       angle1=np.radians(10), angle2=np.radians(10))
        eta_rel_dist = QuadraticBezier(0.84, 0.84, stage_num=comp_stage_num,
                                       angle1=np.radians(3), angle2=np.radians(3))
        c1_a_rel_dist = QuadraticBezier(0.4, 0.4, stage_num=comp_stage_num,
                                       angle1=np.radians(10), angle2=np.radians(10))
        self.comp = Compressor(
            work_fluid=Air(),
            stage_num=comp_stage_num,
            const_diam_par=0.5,
            p0_stag=1e5,
            T0_stag=288,
            G=28,
            n=9e3,
            H_t_rel_arr=H_t_rel_dist.get_array(),
            eta_ad_stag_arr=eta_rel_dist.get_array(),
            R_av_arr=[0.5 for _ in range(comp_stage_num)],
            k_h_arr=[1.0 for _ in range(comp_stage_num)],
            c1_a_rel_arr=c1_a_rel_dist.get_array(),
            h_rk_rel_arr=[3.5 for _ in range(comp_stage_num)],
            h_na_rel_arr=[3.5 for _ in range(comp_stage_num)],
            delta_a_rk_rel_arr=[0.4 for _ in range(comp_stage_num)],
            delta_a_na_rel_arr=[0.4 for _ in range(comp_stage_num)],
            d1_in_rel1=0.5,
            zeta_inlet=0.04, zeta_outlet=0.03,
            c11_init=200, precision=0.0001
        )
        self.comp.compute()
        self.comp_model = CompressorModel(
            characteristics=From16To18Pi(),
            T_stag_in=self.comp.T0_stag,
            p_stag_in=self.comp.p0_stag,
            pi_c_stag_norm_rel=1,
            n_norm_rel=1,
            T_stag_in_nom=self.comp.T0_stag,
            p_stag_in_nom=self.comp.p0_stag,
            G_in_nom=self.comp.G,
            eta_c_stag_nom=self.comp.eta_c_stag,
            n_nom=self.comp.n,
            pi_c_stag_nom=self.comp.pi_c_stag,
            precision=0.0001
        )

    def test_comb_chamber(self):
        self.comb_chamber.compute()
        I_stag_in = self.comb_chamber.work_fluid_in.get_specific_enthalpy(
            self.comb_chamber.T_stag_in, alpha=self.comb_chamber.alpha_in
        ) * self.comb_chamber.G_in
        I_stag_out = self.comb_chamber.work_fluid_out.get_specific_enthalpy(
            self.comb_chamber.T_stag_out, alpha=self.comb_chamber.alpha_out
        ) * self.comb_chamber.G_out
        I_fuel = self.comb_chamber.fuel.get_specific_enthalpy(
            self.comb_chamber.T_fuel, p=self.comb_chamber.p_fuel
        ) * self.comb_chamber.G_fuel
        I_comb = self.comb_chamber.work_fluid_out.Q_n * self.comb_chamber.eta_comb * self.comb_chamber.G_fuel
        q1 = I_stag_in + I_comb + I_fuel
        q2 = I_stag_out
        q_res = abs(q1 - q2) / q1
        self.assertAlmostEqual(q_res, 0, places=3)

    def test_outlet_turbine(self):
        self.turb_model.compute()
        q_out = gd.q(self.turb_model.lam_out, self.turb_model.k_out)
        G_out = q_out * gd.m(self.turb_model.k_out) * self.turb_model.F_out * self.turb_model.p_stag_out / np.sqrt(
            self.turb_model.work_fluid.R * self.turb_model.T_stag_out
        )
        G_res = abs(self.turb_model.G_out - G_out) / self.turb_model.G_out
        self.assertAlmostEqual(G_res, 0, places=3)
        c_out_res = abs(self.turbine.last.c2_a - self.turb_model.c_out) / self.turbine.last.c2_a
        self.assertAlmostEqual(c_out_res, 0, places=1)
        L_res = abs(self.turbine.L_t_sum - abs(self.turb_model.L)) / self.turbine.L_t_sum
        self.assertAlmostEqual(L_res, 0, places=2)
        H_res = abs(self.turbine.H_t_stag - abs(self.turb_model.H_stag)) / self.turbine.H_t_stag
        self.assertAlmostEqual(H_res, 0, places=2)

    def test_compressor(self):
        self.comp_model.compute()
        eta_res = abs(self.comp_model.eta_c_stag_nom - self.comp_model.eta_c_stag) / self.comp_model.eta_c_stag_nom
        pi_c_res = abs(self.comp_model.pi_c_stag - self.comp_model.pi_c_stag_nom) / self.comp_model.pi_c_stag_nom
        G_res = abs(self.comp_model.G_in - self.comp_model.G_in_nom) / self.comp_model.G_in_nom
        n_res = abs(self.comp_model.n - self.comp_model.n_nom) / self.comp_model.n_nom
        self.assertAlmostEqual(eta_res, 0, places=2)
        self.assertAlmostEqual(pi_c_res, 0, places=2)
        self.assertAlmostEqual(G_res, 0, places=2)
        self.assertAlmostEqual(n_res, 0, places=2)
        L_c = self.comp.c_p_av * (self.comp.last.T3_stag - self.comp.first.T1_stag)
        L_res = abs(self.comp_model.L - L_c) / L_c
        self.assertAlmostEqual(L_res, 0, places=3)
        H_c_stag = self.comp.c_p_av * self.comp.T0_stag * (
                self.comp.pi_c_stag ** ((self.comp.k_av - 1) / self.comp.k_av) - 1
        )
        H_c_res = abs(self.comp_model.H_stag - H_c_stag) / H_c_stag
        self.assertAlmostEqual(H_c_res, 0, places=2)

