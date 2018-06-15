import unittest
from .storage import FrequencyBranch, From16To18Pi
from .interp import Interp2D


class TestsBranch(unittest.TestCase):
    def setUp(self):
        self.branch = FrequencyBranch(
            freq_norm_rel=1.00,
            pi_c_stag_rel=[0.43, 0.57, 0.64, 0.725, 0.77, 0.88, 0.94, 1.00, 1.09, 1.13, 1.15],
            G_norm_rel=[1.019, 1.016, 1.013, 1.011, 1.009, 1.006, 1.004, 1.000, 0.991, 0.985, 0.978],
            eta_c_stag_rel=[0.9, 0.92, 0.94, 0.96, 0.98, 1.00, 1.003, 1.0, 0.98, 0.96, 0.94],
            G_spline_deg=3,
            eta_c_spline_deg=2
            )

    def test_plot_branch_without_computing_opt_values(self):
        self.branch.plot()

    def test_plot_branch_with_computing(self):
        self.branch.compute()
        self.branch.plot()


class TestInterp(unittest.TestCase):

    def test_dimension_checking(self):
        interp1 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertTrue(interp1.check_dimension())
        interp2 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2, 4]],
            y_branches=[[1, 2, 3, 5], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5, 6, 4]]
        )
        self.assertFalse(interp2.check_dimension())

    def test_x_values_checking(self):
        interp1 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertTrue(interp1.check_x_values())
        interp2 = Interp2D(
            x_branches=[[1, 1, 2], [2, 1, 2]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertFalse(interp2.check_x_values())

    def test_branches_arrangement_checking(self):
        interp1 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertTrue(interp1.check_branches_arrangement())
        interp2 = Interp2D(
            x_branches=[[2, 2, 2], [1, 1, 1]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertFalse(interp2.check_branches_arrangement())

    def test_points_arrangement_checking(self):
        interp1 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2]],
            y_branches=[[1, 2, 3], [2, 3, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertTrue(interp1.check_points_arrangement())
        interp2 = Interp2D(
            x_branches=[[1, 1, 1], [2, 2, 2]],
            y_branches=[[1, 2, 1], [2, 1, 4]],
            z_branches=[[4, 2, 6], [4, 6, 5]]
        )
        self.assertFalse(interp2.check_points_arrangement())

    def test_getting_value_from_branch(self):
        value1 = Interp2D.get_value_from_branch(3.5, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        value2 = Interp2D.get_value_from_branch(6, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        value3 = Interp2D.get_value_from_branch(-1, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        self.assertEqual(4.5, value1)
        self.assertEqual(7, value2)
        self.assertEqual(0, value3)


class TestCharacteristic(unittest.TestCase):
    def setUp(self):
        self.charact = From16To18Pi(
            grid_pnt_num=20,
            extend=0.4
        )
        self.charact.compute()

    def test_plot_grid_3d(self):
        self.charact.plot_grid_3d()

    def test_plot_grid_2d(self):
        self.charact.plot_grid_2d()

    def test_branches_plot(self):
        self.charact.plot_branches_2d(frequency_int=(0.69, 1.12), branch_num=10)

    def test_nom_point(self):
        eta_nom = self.charact.get_eta_c_stag_rel(1, 1)
        eta_res = abs(eta_nom - 1) / 1
        G_nom = self.charact.get_G_norm_rel(1, 1)
        G_res = abs(G_nom - 1) / 1
        self.assertAlmostEqual(eta_res, 0, places=2)
        self.assertAlmostEqual(G_res, 0, places=2)
