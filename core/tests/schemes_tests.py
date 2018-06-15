import unittest
from ..schemes import TwoShaftGeneratorVar1
from ..compressor_characteristics.storage import From16To18Pi
from compressor.average_streamline.compressor import Compressor as CompAvLine
from compressor.average_streamline.dist_tools import QuadraticBezier
from turbine.average_streamline.turbine import TurbineType
from turbine.average_streamline.turbine import Turbine as TurbAvLine
from gas_turbine_cycle.core.solver import NetworkSolver
from gas_turbine_cycle.core.turbine_lib import Compressor, Sink, Atmosphere, Load, Source, Turbine, \
    CombustionChamber, Inlet, Outlet
from gas_turbine_cycle.gases import Air, NaturalGasCombustionProducts
from gas_turbine_cycle.fuels import NaturalGas
import numpy as np


def compute_cycle(eta_stag_p_c=0.88, eta_stag_p_ct=9.9, eta_stag_p_pt=0.9):
    precision = 0.0001
    atm = Atmosphere()
    inlet = Inlet()
    comp = Compressor(pi_c=17, eta_stag_p=eta_stag_p_c, precision=precision)
    sink = Sink(g_cooling=0, g_outflow=0.03)
    comb_chamber = CombustionChamber(T_gas=1523, precision=precision, alpha_out_init=2, fuel=NaturalGas(),
                                     T_fuel=320, delta_p_fuel=5e5)
    comp_turbine = Turbine(eta_stag_p=eta_stag_p_ct, eta_m=0.99, precision=precision)
    source = Source(g_return=0)
    power_turbine = Turbine(eta_stag_p=eta_stag_p_pt, eta_m=0.99, eta_r=0.99,
                            precision=precision, p_stag_out_init=1e5)
    outlet = Outlet()
    load = Load(power=16e6)
    comp_turb_zero_load = Load(power=0)
    power_turb_zero_load = Load(power=0)

    solver = NetworkSolver(
        unit_arr=[
            atm, inlet, comp, sink, comb_chamber, comp_turbine, source, power_turbine, outlet, load,
            comp_turb_zero_load, power_turb_zero_load
        ],
        relax_coef=1,
        precision=precision,
        max_iter_number=100,
        cold_work_fluid=Air(),
        hot_work_fluid=NaturalGasCombustionProducts()
    )
    solver.create_gas_dynamic_connection(atm, inlet)
    solver.create_gas_dynamic_connection(inlet, comp)
    solver.create_gas_dynamic_connection(comp, sink)
    solver.create_gas_dynamic_connection(sink, comb_chamber)
    solver.create_gas_dynamic_connection(comb_chamber, comp_turbine)
    solver.create_gas_dynamic_connection(comp_turbine, source)
    solver.create_gas_dynamic_connection(source, power_turbine)
    solver.create_gas_dynamic_connection(power_turbine, outlet)
    solver.create_static_gas_dynamic_connection(outlet, atm)
    solver.create_mechanical_connection(comp_turbine, comp, comp_turb_zero_load)
    solver.create_mechanical_connection(power_turbine, load, power_turb_zero_load)
    solver.solve()
    G_air = load.power / power_turbine.gen_labour1
    G_comp_turb = G_air * comp_turbine.g_in
    G_power_turb = G_air * power_turbine.g_in
    G_fuel = G_air * comb_chamber.g_fuel_prime * comb_chamber.g_in
    eta_e = power_turbine.gen_labour1 / (comb_chamber.work_fluid_out.Q_n * comb_chamber.g_fuel_prime *
                                         comb_chamber.g_in)
    return solver, G_air, G_comp_turb, G_power_turb, G_fuel, eta_e


def get_compressor(p0_stag, T0_stag, G, n, stage_num, H_t_rel1, H_t_rel_delta, c1_a_rel1, c1_a_rel_delta) -> CompAvLine:
    H_t_rel_dist = QuadraticBezier(H_t_rel1, H_t_rel1 - H_t_rel_delta, stage_num=stage_num,
                                   angle1=np.radians(10), angle2=np.radians(10))
    eta_rel_dist = QuadraticBezier(0.86, 0.86, stage_num=stage_num,
                                   angle1=np.radians(3), angle2=np.radians(3))
    c1_a_rel_dist = QuadraticBezier(c1_a_rel1, c1_a_rel1 - c1_a_rel_delta, stage_num=stage_num,
                                    angle1=np.radians(10), angle2=np.radians(10))
    comp = CompAvLine(
        work_fluid=Air(),
        stage_num=stage_num,
        const_diam_par_arr=[0.5 for _ in range(stage_num)],
        p0_stag=p0_stag,
        T0_stag=T0_stag,
        G=G,
        n=n,
        H_t_rel_arr=H_t_rel_dist.get_array(),
        eta_ad_stag_arr=eta_rel_dist.get_array(),
        R_av_arr=[0.5 for _ in range(stage_num)],
        k_h_arr=[1.0 for _ in range(stage_num)],
        c1_a_rel_arr=c1_a_rel_dist.get_array(),
        h_rk_rel_arr=[3.5 for _ in range(stage_num)],
        h_na_rel_arr=[3.5 for _ in range(stage_num)],
        delta_a_rk_rel_arr=[0.4 for _ in range(stage_num)],
        delta_a_na_rel_arr=[0.4 for _ in range(stage_num)],
        d1_in_rel1=0.5,
        zeta_inlet=0.04, zeta_outlet=0.03,
        c11_init=200, precision=0.0001
    )
    comp.compute()
    return comp


def get_optimize_compressor(p0_stag, T0_stag, G, n, pi_c_stag, stage_num_arr=[int(i) for i in np.linspace(9, 15, 7)],
                            H_t_rel1_arr=np.linspace(0.2, 0.35, 10), H_t_rel_delta=-0.06,
                            c1_a_rel1_arr=np.linspace(0.45, 0.55, 10), c1_a_rel_delta=0.1):
    res = []
    for stage_num in stage_num_arr:
        for H_t_rel1 in H_t_rel1_arr:
            for c1_a_rel1 in c1_a_rel1_arr:
                comp = get_compressor(p0_stag, T0_stag, G, n, stage_num, H_t_rel1,
                                      H_t_rel_delta, c1_a_rel1, c1_a_rel_delta)
                res.append(comp)
    opt_comp = res[0]
    for comp in res:
        if abs(comp.pi_c_stag - pi_c_stag) < abs(opt_comp.pi_c_stag - pi_c_stag):
            opt_comp = comp
    return opt_comp


def compute_cycle_and_nodes(pi_c_stag=17, stage_num_arr=[int(i) for i in np.linspace(9, 15, 7)],
                            H_t_rel1_arr=np.linspace(0.2, 0.35, 10), H_t_rel_delta=-0.06,
                            c1_a_rel1_arr=np.linspace(0.45, 0.55, 10), c1_a_rel_delta=0.1):
    eta_stag_p_c = 0.87
    eta_stag_p_ct = 0.89
    eta_stag_p_pt = 0.89
    res = 1
    iter_num = 1
    while res >= 0.001:
        print('Iter %s' % iter_num)
        iter_num += 1
        solver, G_air, G_comp_turb, G_power_turb, G_fuel, eta_e = compute_cycle(eta_stag_p_c,
                                                                                eta_stag_p_ct, eta_stag_p_pt)
        n_ct = 11e3
        n_pt = 7.8e3
        inlet: Inlet = solver.get_sorted_unit_list()[1]
        comp_cycle: Compressor = solver.get_sorted_unit_list()[2]
        comp_turbine_cycle: Turbine = solver.get_sorted_unit_list()[5]
        power_turbine_cycle: Turbine = solver.get_sorted_unit_list()[7]
        sink: Sink = solver.get_sorted_unit_list()[3]
        comb_chamber: CombustionChamber = solver.get_sorted_unit_list()[4]
        outlet: Outlet = solver.get_sorted_unit_list()[8]
        load: Load = solver.get_sorted_unit_list()[9]

        comp = get_optimize_compressor(
            comp_cycle.p_stag_in, comp_cycle.T_stag_in, G_air, n_ct, pi_c_stag,
            stage_num_arr, H_t_rel1_arr, H_t_rel_delta, c1_a_rel1_arr, c1_a_rel_delta
        )

        comp_turbine = TurbAvLine(
            turbine_type=TurbineType.WORK,
            stage_number=2,
            T_g_stag=comp_turbine_cycle.T_stag_in,
            p_g_stag=comp_turbine_cycle.p_stag_in,
            G_turbine=G_comp_turb,
            G_fuel=G_fuel,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.15,
            n=n_ct,
            T_t_stag_cycle=comp_turbine_cycle.T_stag_out,
            eta_t_stag_cycle=comp_turbine_cycle.eta_stag,
            alpha11=np.radians(14),
            precision=0.0001,
            eta_m=comp_turbine_cycle.eta_m,
            gamma_in=np.radians(0),
            gamma_out=np.radians(10),
            H01_init=200e3,
            c21_init=250
        )
        comp_turbine.compute_geometry()
        comp_turbine.compute_stages_gas_dynamics()
        comp_turbine.compute_integrate_turbine_parameters()

        power_turbine = TurbAvLine(
            turbine_type=TurbineType.WORK,
            stage_number=2,
            T_g_stag=power_turbine_cycle.T_stag_in,
            p_g_stag=power_turbine_cycle.p_stag_in,
            G_turbine=G_power_turb,
            G_fuel=G_fuel,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.15,
            n=n_pt,
            eta_m=power_turbine_cycle.eta_m,
            T_t_stag_cycle=power_turbine_cycle.T_stag_out,
            eta_t_stag_cycle=power_turbine_cycle.eta_stag,
            alpha11=np.radians(14),
            precision=0.0001,
            gamma_in=np.radians(0),
            gamma_out=np.radians(15),
            H01_init=200e3,
            c21_init=250
        )
        power_turbine.compute_geometry()
        power_turbine.compute_stages_gas_dynamics()
        power_turbine.compute_integrate_turbine_parameters()

        eta_stag_p_c_res = abs(comp.eta_c_stag_p - eta_stag_p_c) / eta_stag_p_c
        eta_stag_p_ct_res = abs(comp_turbine.eta_t_stag_p - eta_stag_p_ct) / eta_stag_p_ct
        eta_stag_p_pt_res = abs(power_turbine.eta_t_stag_p - eta_stag_p_pt) / eta_stag_p_pt

        eta_stag_p_c = comp.eta_c_stag_p
        eta_stag_p_ct = comp_turbine.eta_t_stag_p
        eta_stag_p_pt = power_turbine.eta_t_stag_p

        res = max(eta_stag_p_c_res, eta_stag_p_pt_res, eta_stag_p_ct_res)

    return solver, inlet, comp, sink, comb_chamber, comp_turbine, power_turbine, outlet, load, comp_cycle


class TestUnitsNominalCalculation(unittest.TestCase):
    def setUp(self):
        self.pi_c_stag = 17
        self.solver, self.inlet, self.comp, self.sink, self.comb_chamber, self.comp_turbine, \
        self.power_turbine, self.outlet, self.load, self.comp_cycle = compute_cycle_and_nodes(
            self.pi_c_stag, stage_num_arr=[int(i) for i in np.linspace(16, 22, 7)],
            H_t_rel1_arr=np.linspace(0.12, 0.23, 6), H_t_rel_delta=-0.06,
            c1_a_rel1_arr=np.linspace(0.45, 0.55, 5), c1_a_rel_delta=0.1
        )

    def test_compressor_optimizing(self):
        comp = get_optimize_compressor(
            1e5, 288, 38, 11e3,
            self.pi_c_stag, stage_num_arr=[int(i) for i in np.linspace(17, 23, 7)],
            H_t_rel1_arr=np.linspace(0.16, 0.38, 6), H_t_rel_delta=-0.06,
            c1_a_rel1_arr=np.linspace(0.42, 0.58, 6), c1_a_rel_delta=0.1
        )
        pi_c_res = abs(self.pi_c_stag - comp.pi_c_stag) / self.pi_c_stag
        self.assertAlmostEqual(pi_c_res, 0, places=2)

    def test_compressor_cycle_node_coincidence(self):
        comp_cycle: Compressor = self.solver.get_sorted_unit_list()[2]
        L_c = self.comp.c_p_av * (self.comp.last.T3_stag - self.comp.T0_stag)
        L_c_res = abs(comp_cycle.consumable_labour - L_c) / L_c
        T_out_res = abs(self.comp.last.T3_stag - comp_cycle.T_stag_out) / comp_cycle.T_stag_out
        p_out_res = abs(self.comp.last.p3_stag - comp_cycle.p_stag_out) / comp_cycle.p_stag_out
        self.assertAlmostEqual(T_out_res, 0, places=2)
        self.assertAlmostEqual(p_out_res, 0, places=2)
        self.assertAlmostEqual(L_c_res, 0, places=2)

    def test_comp_turbine_cycle_node_coincidence(self):
        comp_turbine_cycle: Turbine = self.solver.get_sorted_unit_list()[5]
        T_out_res = abs(self.comp_turbine.last.T_st_stag - comp_turbine_cycle.T_stag_out) / comp_turbine_cycle.T_stag_out
        p_out_res = abs(self.comp_turbine.last.p2_stag - comp_turbine_cycle.p_stag_out) / comp_turbine_cycle.p_stag_out
        L_res = abs(self.comp_turbine.L_t_sum - comp_turbine_cycle.total_labour) / comp_turbine_cycle.total_labour
        self.assertAlmostEqual(T_out_res, 0, places=3)
        self.assertAlmostEqual(p_out_res, 0, places=3)
        self.assertAlmostEqual(L_res, 0, places=3)

    def test_power_turbine_cycle_node_coincidence(self):
        power_turbine_cycle: Turbine = self.solver.get_sorted_unit_list()[7]
        T_out_res = abs(self.power_turbine.last.T_st_stag - power_turbine_cycle.T_stag_out) / power_turbine_cycle.T_stag_out
        p_out_res = abs(self.power_turbine.last.p2_stag - power_turbine_cycle.p_stag_out) / power_turbine_cycle.p_stag_out
        L_res = abs(self.power_turbine.L_t_sum - power_turbine_cycle.total_labour) / power_turbine_cycle.total_labour
        self.assertAlmostEqual(T_out_res, 0, places=3)
        self.assertAlmostEqual(p_out_res, 0, places=2)
        self.assertAlmostEqual(L_res, 0, places=3)


class SchemeTests(unittest.TestCase):
    def setUp(self):
        self.pi_c_stag = 17
        self.solver, self.inlet, self.comp, self.sink, self.comb_chamber, self.comp_turbine, \
        self.power_turbine, self.outlet, self.load, self.comp_cycle = compute_cycle_and_nodes(
            self.pi_c_stag, stage_num_arr=[int(i) for i in np.linspace(16, 22, 7)],
            H_t_rel1_arr=np.linspace(0.15, 0.30, 6), H_t_rel_delta=-0.06,
            c1_a_rel1_arr=np.linspace(0.45, 0.55, 5), c1_a_rel_delta=0.1
        )
        self.power_turbine_cycle: Turbine = self.solver.get_sorted_unit_list()[7]
        self.scheme = TwoShaftGeneratorVar1(
            inlet=self.inlet,
            compressor=self.comp_cycle,
            comp_turbine=self.comp_turbine,
            power_turbine=self.power_turbine,
            comb_chamber=self.comb_chamber,
            outlet=self.outlet,
            p_stag_in=self.inlet.p_stag_in,
            T_a=self.inlet.T_stag_in,
            p_a=self.inlet.p_stag_in,
            G_nom=self.comp.G,
            n_nom=self.comp.n,
            g_cool_sum=self.sink.g_cooling + self.sink.g_outflow,
            g_cool_st1=0,
            g_cool_st2=0,
            eta_r=self.power_turbine_cycle.eta_r,
            pi_c_stag_rel_init=0.95,
            n_norm_rel_init=0.95,
            pi_t1_stag_init=1.75,
            pi_t2_stag_init=1.9,
            pi_t3_stag_init=4,
            comp_char=From16To18Pi(),
            outlet_diff_coef=2.5,
            precision=0.0001,
            g_fuel_init=self.comb_chamber.g_fuel_prime * self.comb_chamber.g_in,
            T_stag_in_arr=np.linspace(298, 268, 20),
            N_e_max=16.4e6,
            T_g_stag=self.comb_chamber.T_stag_out
        )

    def test_compressor_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.comp_model.p_stag_in_nom, self.inlet.p_stag_in)
        self.assertEqual(self.scheme.comp_model.T_stag_in_nom, self.inlet.T_stag_in)
        self.assertEqual(self.scheme.comp_model.n_nom, self.comp.n)
        self.assertEqual(self.scheme.comp_model.eta_c_stag_nom, self.comp_cycle.eta_stag)
        self.assertEqual(self.scheme.comp_model.G_in_nom, self.comp.G)
        self.assertEqual(self.scheme.comp_model.pi_c_stag_nom, self.comp_cycle.pi_c)

    def test_sink_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.sink_model.g_cool, self.sink.g_cooling + self.sink.g_outflow)

    def test_comp_turbine_first_stage_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.comp_turb_st1_model.T_stag_in_nom, self.comp_turbine[0].T0_stag)
        self.assertEqual(self.scheme.comp_turb_st1_model.p_stag_in_nom, self.comp_turbine[0].p0_stag)
        self.assertEqual(self.scheme.comp_turb_st1_model.pi_t_stag_nom, self.comp_turbine[0].pi_stag)
        self.assertEqual(self.scheme.comp_turb_st1_model.eta_t_stag_nom, self.comp_turbine[0].eta_t_stag)
        self.assertEqual(self.scheme.comp_turb_st1_model.G_in_nom, self.comp_turbine[0].G_stage_in)
        self.assertEqual(self.scheme.comp_turb_st1_model.eta_m, self.comp_turbine.eta_m)

    def test_source_first_stage_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.source_st1_model.T_cool, self.comp_turbine[0].T_cool)

    def test_comp_turbine_second_stage_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.comp_turb_st2_model.T_stag_in_nom, self.comp_turbine[1].T0_stag)
        self.assertEqual(self.scheme.comp_turb_st2_model.p_stag_in_nom, self.comp_turbine[1].p0_stag)
        self.assertEqual(self.scheme.comp_turb_st2_model.pi_t_stag_nom, self.comp_turbine[1].pi_stag)
        self.assertEqual(self.scheme.comp_turb_st2_model.eta_t_stag_nom, self.comp_turbine[1].eta_t_stag)
        self.assertEqual(self.scheme.comp_turb_st2_model.G_in_nom, self.comp_turbine[1].G_stage_in)
        self.assertEqual(self.scheme.comp_turb_st2_model.eta_m, self.comp_turbine.eta_m)

    def test_source_second_stage_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.source_st2_model.T_cool, self.comp_turbine[1].T_cool)

    def test_power_turbine_nominal_parameters_setting(self):
        self.scheme.init_models_with_nominal_params()
        self.assertEqual(self.scheme.power_turb_model.T_stag_in_nom, self.power_turbine.T_g_stag)
        self.assertEqual(self.scheme.power_turb_model.p_stag_in_nom, self.power_turbine.p_g_stag)
        self.assertEqual(self.scheme.power_turb_model.G_in_nom, self.power_turbine.G_turbine)
        self.assertEqual(self.scheme.power_turb_model.eta_t_stag_nom, self.power_turbine.eta_t_stag)
        self.assertEqual(self.scheme.power_turb_model.pi_t_stag_nom, self.power_turbine.pi_t_stag)
        self.assertEqual(self.scheme.power_turb_model.F_out, self.power_turbine.geom.last.A2)
        self.assertEqual(self.scheme.power_turb_model.eta_m, self.power_turbine.eta_m)

    def test_parameters_transfer(self):
        self.scheme.init_models_with_nominal_params()
        self.scheme.compute(1, 1, self.comp_turbine[0].pi_stag, self.comp_turbine[1].pi_stag,
                            self.power_turbine.pi_t_stag,
                            self.comb_chamber.g_fuel_prime * self.comb_chamber.g_in,
                            self.inlet.T_stag_in)
        self.assertEqual(self.scheme.comp_model.T_stag_out, self.scheme.sink_model.T_stag_in)
        self.assertEqual(self.scheme.comp_model.p_stag_out, self.scheme.sink_model.p_stag_in)
        self.assertEqual(self.scheme.comp_model.G_out, self.scheme.sink_model.G_in)
        self.assertEqual(self.scheme.comp_model.G_fuel_out, self.scheme.sink_model.G_fuel_in)

        self.assertEqual(self.scheme.sink_model.T_stag_out, self.scheme.comb_chamber_model.T_stag_in)
        self.assertEqual(self.scheme.sink_model.p_stag_out, self.scheme.comb_chamber_model.p_stag_in)
        self.assertEqual(self.scheme.sink_model.G_out, self.scheme.comb_chamber_model.G_in)
        self.assertEqual(self.scheme.sink_model.G_fuel_out, self.scheme.comb_chamber_model.G_fuel_in)
        self.assertEqual(self.scheme.sink_model.G_c1_in, self.scheme.comp_model.G_in)

        self.assertEqual(self.scheme.comb_chamber_model.T_stag_out, self.scheme.comp_turb_st1_model.T_stag_in)
        self.assertEqual(self.scheme.comb_chamber_model.p_stag_out, self.scheme.comp_turb_st1_model.p_stag_in)
        self.assertEqual(self.scheme.comb_chamber_model.G_out, self.scheme.comp_turb_st1_model.G_in)
        self.assertEqual(self.scheme.comb_chamber_model.G_fuel_out, self.scheme.comp_turb_st1_model.G_fuel_in)
        self.assertEqual(self.scheme.comb_chamber_model.G_c1_in, self.scheme.comp_model.G_in)

        self.assertEqual(self.scheme.comp_turb_st1_model.T_stag_out, self.scheme.source_st1_model.T_stag_in)
        self.assertEqual(self.scheme.comp_turb_st1_model.p_stag_out, self.scheme.source_st1_model.p_stag_in)
        self.assertEqual(self.scheme.comp_turb_st1_model.G_out, self.scheme.source_st1_model.G_in)
        self.assertEqual(self.scheme.comp_turb_st1_model.G_fuel_out, self.scheme.source_st1_model.G_fuel_in)

        self.assertEqual(self.scheme.source_st1_model.T_stag_out, self.scheme.comp_turb_st2_model.T_stag_in)
        self.assertEqual(self.scheme.source_st1_model.p_stag_out, self.scheme.comp_turb_st2_model.p_stag_in)
        self.assertEqual(self.scheme.source_st1_model.G_out, self.scheme.comp_turb_st2_model.G_in)
        self.assertEqual(self.scheme.source_st1_model.G_fuel_out, self.scheme.comp_turb_st2_model.G_fuel_in)
        self.assertEqual(self.scheme.source_st1_model.G_c1_in, self.scheme.comp_model.G_in)

        self.assertEqual(self.scheme.comp_turb_st2_model.T_stag_out, self.scheme.source_st2_model.T_stag_in)
        self.assertEqual(self.scheme.comp_turb_st2_model.p_stag_out, self.scheme.source_st2_model.p_stag_in)
        self.assertEqual(self.scheme.comp_turb_st2_model.G_out, self.scheme.source_st2_model.G_in)
        self.assertEqual(self.scheme.comp_turb_st2_model.G_fuel_out, self.scheme.source_st2_model.G_fuel_in)

        self.assertEqual(self.scheme.source_st2_model.T_stag_out, self.scheme.power_turb_model.T_stag_in)
        self.assertEqual(self.scheme.source_st2_model.p_stag_out, self.scheme.power_turb_model.p_stag_in)
        self.assertEqual(self.scheme.source_st2_model.G_out, self.scheme.power_turb_model.G_in)
        self.assertEqual(self.scheme.source_st2_model.G_fuel_out, self.scheme.power_turb_model.G_fuel_in)
        self.assertEqual(self.scheme.source_st2_model.G_c1_in, self.scheme.comp_model.G_in)

        self.assertEqual(self.scheme.power_turb_model.T_stag_out, self.scheme.outlet_model.T_stag_in)
        self.assertEqual(self.scheme.power_turb_model.p_stag_out, self.scheme.outlet_model.p_stag_in)
        self.assertEqual(self.scheme.power_turb_model.G_out, self.scheme.outlet_model.G_in)
        self.assertEqual(self.scheme.power_turb_model.G_fuel_out, self.scheme.outlet_model.G_fuel_in)
        self.assertEqual(self.scheme.power_turb_model.F_out, self.scheme.outlet_model.F_in)
        self.assertEqual(self.scheme.power_turb_model.c_out, self.scheme.outlet_model.c_in)

    def test_nominal_mode(self):
        self.scheme.init_models_with_nominal_params()
        self.scheme.compute(1, 1, self.comp_turbine[0].pi_stag, self.comp_turbine[1].pi_stag,
                            self.power_turbine.pi_t_stag,
                            self.comb_chamber.g_fuel_prime * self.comb_chamber.g_in,
                            self.inlet.T_stag_in)

        T_comp_out_res = abs(self.scheme.comp_model.T_stag_out - self.comp_cycle.T_stag_out) / self.comp_cycle.T_stag_out
        T_g_res = abs(
            self.scheme.comb_chamber_model.T_stag_out - self.comb_chamber.T_stag_out) / self.comb_chamber.T_stag_out
        T_st1_res = abs(
            self.scheme.source_st1_model.T_stag_out - self.comp_turbine[0].T_st_stag) / self.comp_turbine[0].T_st_stag
        T_st2_res = abs(
            self.scheme.source_st2_model.T_stag_out - self.comp_turbine[1].T_st_stag) / self.comp_turbine[1].T_st_stag
        T_pt_res = abs(
            self.scheme.power_turb_model.T_stag_out - self.power_turbine.last.T_st_stag) / self.power_turbine.last.T_st_stag

        p_comp_out_res = abs(self.scheme.comp_model.p_stag_out - self.comp_cycle.p_stag_out) / self.comp_cycle.p_stag_out
        p_g_res = abs(
            self.scheme.comb_chamber_model.p_stag_out - self.comb_chamber.p_stag_out) / self.comb_chamber.p_stag_out
        p_st1_res = abs(
            self.scheme.source_st1_model.p_stag_out - self.comp_turbine[0].p2_stag) / self.comp_turbine[0].p2_stag
        p_st2_res = abs(
            self.scheme.source_st2_model.p_stag_out - self.comp_turbine[1].p2_stag) / self.comp_turbine[1].p2_stag
        p_pt_res = abs(
            self.scheme.power_turb_model.p_stag_out - self.power_turbine.last.p2_stag) / self.power_turbine.last.p2_stag

        L_c = self.comp_cycle.consumable_labour
        L_c_res = abs(
            self.scheme.comp_model.L - L_c) / L_c
        L_st1_res = abs(
            self.scheme.comp_turb_st1_model.L + self.comp_turbine[0].L_t) / self.comp_turbine[0].L_t
        L_st2_res = abs(
            self.scheme.comp_turb_st2_model.L + self.comp_turbine[1].L_t) / self.comp_turbine[1].L_t
        L_pt_res = abs(
            self.scheme.power_turb_model.L + self.power_turbine.L_t_sum) / self.power_turbine.L_t_sum

        eta_c_res = abs(
            self.scheme.comp_model.eta_c_stag - self.comp_cycle.eta_stag) / self.comp_cycle.eta_stag
        eta_st1_res = abs(
            self.scheme.comp_turb_st1_model.eta_t_stag - self.comp_turbine[0].eta_t_stag) / self.comp_turbine[0].eta_t_stag
        eta_st2_res = abs(
            self.scheme.comp_turb_st2_model.eta_t_stag - self.comp_turbine[1].eta_t_stag) / self.comp_turbine[1].eta_t_stag
        eta_pt_res = abs(
            self.scheme.power_turb_model.eta_t_stag - self.power_turbine.eta_t_stag) / self.power_turbine.eta_t_stag

        G_c_res = abs(
            self.scheme.comp_model.G_in - self.comp.G) / self.comp.G
        G_fuel_res = abs(
            self.scheme.comb_chamber_model.G_fuel - self.comp_turbine.G_fuel) / self.comp_turbine.G_fuel
        G_st1_res = abs(
            self.scheme.comp_turb_st1_model.G_in_char - self.comp_turbine[0].G_stage_in) / self.comp_turbine[
                          0].G_stage_in
        G_st2_res = abs(
            self.scheme.comp_turb_st2_model.G_in_char - self.comp_turbine[1].G_stage_in) / self.comp_turbine[
                          1].G_stage_in
        G_pt_res = abs(
            self.scheme.power_turb_model.G_in_char - self.power_turbine.G_turbine) / self.power_turbine.G_turbine

        comp_power = self.scheme.comp_model.N
        turb_power = self.scheme.comp_turb_st1_model.N + self.scheme.comp_turb_st2_model.N
        work_balance_res = abs(comp_power + turb_power) / comp_power

        N_pt_res = abs(self.load.power + self.scheme.power_turb_model.N) / self.load.power

        self.assertAlmostEqual(T_comp_out_res, 0, places=2)
        self.assertAlmostEqual(T_g_res, 0, places=2)
        self.assertAlmostEqual(T_st1_res, 0, places=2)
        self.assertAlmostEqual(T_st2_res, 0, places=2)
        self.assertAlmostEqual(T_pt_res, 0, places=2)
        self.assertAlmostEqual(p_comp_out_res, 0, places=2)
        self.assertAlmostEqual(p_g_res, 0, places=2)
        self.assertAlmostEqual(p_st1_res, 0, places=2)
        self.assertAlmostEqual(p_st2_res, 0, places=2)
        self.assertAlmostEqual(p_pt_res, 0, places=2)
        self.assertAlmostEqual(L_c_res, 0, places=2)
        self.assertAlmostEqual(L_st1_res, 0, places=2)
        self.assertAlmostEqual(L_st2_res, 0, places=2)
        self.assertAlmostEqual(L_pt_res, 0, places=2)
        self.assertAlmostEqual(eta_c_res, 0, places=2)
        self.assertAlmostEqual(eta_st1_res, 0, places=2)
        self.assertAlmostEqual(eta_st2_res, 0, places=2)
        self.assertAlmostEqual(eta_pt_res, 0, places=2)
        self.assertAlmostEqual(G_c_res, 0, places=2)
        self.assertAlmostEqual(G_fuel_res, 0, places=2)
        self.assertAlmostEqual(G_st1_res, 0, places=2)
        self.assertAlmostEqual(G_st2_res, 0, places=2)
        self.assertAlmostEqual(G_pt_res, 0, places=2)
        self.assertAlmostEqual(work_balance_res, 0, places=2)
        self.assertLess(N_pt_res, 0.01)

    def test_solving(self):
        self.scheme.solve()
        print(self.scheme.modes_params)
        self.scheme.comp_model.characteristics.plot_modes_line(
            pi_c_stag_rel_arr=self.scheme.modes_params['pi_c_stag_rel'],
            G_norm_rel_arr=self.scheme.modes_params['G_in_norm_rel'],
            eta_c_stag_rel_arr=self.scheme.modes_params['eta_c_stag_rel'],
            figsize=(8, 7)
        )
        self.scheme.plot_inlet_temp_plot(
            T_stag_in_arr=self.scheme.modes_params['T_stag_in'] - 273,
            value_arr=self.scheme.modes_params['N_e'] / 1e6,
            value_label=r'$N_e,\ МВт$',
            figsize=(7, 5)
        )
        self.scheme.plot_inlet_temp_plot(
            T_stag_in_arr=self.scheme.modes_params['T_stag_in'] - 273,
            value_arr=self.scheme.modes_params['g_fuel'],
            value_label=r'$g_т$',
            figsize=(7, 5)
        )
        self.scheme.plot_inlet_temp_plot(
            T_stag_in_arr=self.scheme.modes_params['T_stag_in'] - 273,
            value_arr=self.scheme.modes_params['T_out'] - 273,
            value_label=r'$T_{вых}^*,\ С$',
            figsize=(7, 5)
        )
        self.scheme.plot_inlet_temp_plot(
            T_stag_in_arr=self.scheme.modes_params['T_stag_in'] - 273,
            value_arr=self.scheme.modes_params['G_out'],
            value_label=r'$G_{вых},\ кг/с$',
            figsize=(7, 5)
        )
        self.scheme.plot_inlet_temp_plot(
            T_stag_in_arr=self.scheme.modes_params['T_stag_in'] - 273,
            value_arr=self.scheme.modes_params['eta_e'],
            value_label=r'$\eta_{e}$',
            figsize=(7, 5)
        )

