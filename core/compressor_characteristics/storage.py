from .tools import Characteristics, FrequencyBranch


class From16To18Pi(Characteristics):
    def __init__(self, extend: float=0.2, grid_pnt_num: int=50):
        Characteristics.__init__(self, extend, grid_pnt_num)
        self.branches = [
            FrequencyBranch(
                freq_norm_rel=0.7,
                pi_c_stag_rel=[0.05, 0.105, 0.145, 0.167, 0.195],
                G_norm_rel=[0.345, 0.33, 0.32, 0.315, 0.3],
                eta_c_stag_rel=[0.892, 0.908, 0.916, 0.920, 0.923],
                G_spline_deg=2,
                eta_c_spline_deg=2,
            ),
            FrequencyBranch(
                freq_norm_rel=0.8,
                pi_c_stag_rel=[0.065, 0.18, 0.22, 0.255, 0.3, 0.36],
                G_norm_rel=[0.467, 0.457, 0.454, 0.450, 0.442, 0.435],
                eta_c_stag_rel=[0.895, 0.920, 0.93, 0.94, 0.947, 0.943],
                G_spline_deg=3,
                eta_c_spline_deg=2,
            ),
            FrequencyBranch(
                freq_norm_rel=0.85,
                pi_c_stag_rel=[0.133, 0.23, 0.255, 0.28, 0.33, 0.38, 0.46, 0.58],
                G_norm_rel=[0.603, 0.5985, 0.5975, 0.5970, 0.594, 0.591, 0.58, 0.56],
                eta_c_stag_rel=[0.90, 0.92, 0.93, 0.937, 0.95, 0.96, 0.98, 0.97],
                G_spline_deg=3,
                eta_c_spline_deg=2,
            ),
            FrequencyBranch(
                freq_norm_rel=0.9,
                pi_c_stag_rel=[0.28, 0.335, 0.36, 0.39, 0.465, 0.495, 0.555, 0.65, 0.74, 0.77],
                G_norm_rel=[0.785, 0.784, 0.7835, 0.7825, 0.780, 0.778, 0.776, 0.77, 0.75, 0.735],
                eta_c_stag_rel=[0.91, 0.92, 0.93, 0.94, 0.96, 0.98, 1.0, 1.008, 1.003, 0.988],
                G_spline_deg=3,
                eta_c_spline_deg=4,
            ),
            FrequencyBranch(
                freq_norm_rel=0.95,
                pi_c_stag_rel=[0.38, 0.44, 0.52, 0.58, 0.63, 0.69, 0.75, 0.815, 0.875, 0.935, 0.975],
                G_norm_rel=[0.925, 0.922, 0.917, 0.913, 0.909, 0.904, 0.902, 0.898, 0.895, 0.89, 0.875],
                eta_c_stag_rel=[0.905, 0.92, 0.94, 0.96, 0.98, 1.00, 1.01, 1.016, 1.012, 1.006, 0.989],
                G_spline_deg=3,
                eta_c_spline_deg=3,
            ),
            FrequencyBranch(
                freq_norm_rel=1.00,
                pi_c_stag_rel=[0.43, 0.57, 0.64, 0.725, 0.77, 0.88, 0.94, 1.00, 1.09, 1.13, 1.15],
                G_norm_rel=[1.019, 1.016, 1.013, 1.011, 1.009, 1.005, 1.002, 0.998, 0.991, 0.985, 0.978],
                eta_c_stag_rel=[0.9, 0.92, 0.94, 0.96, 0.98, 1.00, 1.003, 1.0, 0.98, 0.96, 0.94],
                G_spline_deg=3,
                eta_c_spline_deg=2,
            ),
            FrequencyBranch(
                freq_norm_rel=1.1,
                pi_c_stag_rel=[0.66, 0.8, 0.98, 1.185, 1.25, 1.35],
                G_norm_rel=[1.135, 1.1325, 1.1295, 1.125, 1.117, 1.105],
                eta_c_stag_rel=[0.885, 0.9, 0.92, 0.943, 0.942, 0.92],
                G_spline_deg=3,
                eta_c_spline_deg=2,
            )
        ]
