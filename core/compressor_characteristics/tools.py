import typing
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .interp import Interp2D
from abc import ABCMeta


class FrequencyBranch:
    def __init__(self, freq_norm_rel: float,
                 pi_c_stag_rel: typing.List[float],
                 G_norm_rel: typing.List[float],
                 eta_c_stag_rel: typing.List[float],
                 G_spline_deg: int=3,
                 eta_c_spline_deg: int=3,
                 spline_smooth_factor: float = 2.,
                 spline_pnt_num: int=30
                 ):
        self.freq_norm_rel = freq_norm_rel
        self.pi_c_stag_rel = pi_c_stag_rel
        self.G_norm_rel = G_norm_rel
        self.eta_c_stag_rel = eta_c_stag_rel
        assert len(pi_c_stag_rel) == len(G_norm_rel) == len(eta_c_stag_rel), \
            'In frequency branch %s parameters lists must have the same length' % freq_norm_rel
        self.G_spline_deg = G_spline_deg
        self.eta_c_spline_deg = eta_c_spline_deg
        self.spline_smooth_factor = spline_smooth_factor
        self.spline_pnt_num = spline_pnt_num
        self.eta_c_splrep = None
        self.G_splrep = None
        self.norm_values = None

    def get_G_spline_points(self, pi_c_min: float, pi_c_max: float, pnt_num: int):
        assert self.G_splrep is not None, 'Spline must be computed.'
        pi_c_stag_rel_norm_new = np.linspace(pi_c_min, pi_c_max, pnt_num)
        G_rel_norm_new = splev(pi_c_stag_rel_norm_new, self.G_splrep)
        return pi_c_stag_rel_norm_new, G_rel_norm_new

    def get_eta_c_stag_spline_points(self, G_min: float, G_max: float, pnt_num: int):
        assert self.eta_c_splrep is not None, 'Spline must be computed.'
        G_rel_norm_new = np.linspace(G_min, G_max, pnt_num)
        eta_c_stag_rel_norm_new = splev(G_rel_norm_new, self.eta_c_splrep)
        return G_rel_norm_new, eta_c_stag_rel_norm_new

    def compute(self):
        self.G_splrep = splrep(self.pi_c_stag_rel, self.G_norm_rel,
                               k=self.G_spline_deg, s=self.spline_smooth_factor)
        self.G_norm_rel.reverse()
        self.eta_c_stag_rel.reverse()
        self.eta_c_splrep = splrep(self.G_norm_rel, self.eta_c_stag_rel,
                                   k=self.eta_c_spline_deg, s=self.spline_smooth_factor)
        self.G_norm_rel.reverse()
        self.eta_c_stag_rel.reverse()

    def plot(self, figsize=(7, 10)):
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.scatter(self.G_norm_rel, self.pi_c_stag_rel, s=40, c='orange')
        if self.G_splrep:
            pi_c, G = self.get_G_spline_points(min(self.pi_c_stag_rel),
                                               max(self.pi_c_stag_rel),
                                               self.spline_pnt_num)
            plt.plot(G, pi_c, lw=1, ls='--', color='orange')
        plt.grid()
        plt.xlim(round(min(self.G_norm_rel), 1) - 0.05, round(max(self.G_norm_rel), 1) + 0.05)
        plt.ylim(round(min(self.pi_c_stag_rel), 1) - 0.05, round(max(self.pi_c_stag_rel), 1) + 0.05)
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\pi}_к$', fontsize=10)
        plt.subplot(212)
        plt.scatter(self.G_norm_rel, self.eta_c_stag_rel, s=40, c='blue')
        if self.eta_c_splrep:
            G, eta_c = self.get_eta_c_stag_spline_points(min(self.G_norm_rel),
                                                         max(self.G_norm_rel),
                                                         self.spline_pnt_num)
            plt.plot(G, eta_c, lw=1, ls='--', color='blue')
        plt.grid()
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\eta}_к^*$', fontsize=10)
        plt.xlim(round(min(self.G_norm_rel), 1) - 0.05, round(max(self.G_norm_rel), 1) + 0.05)
        plt.ylim(round(min(self.eta_c_stag_rel), 1) - 0.05, round(max(self.eta_c_stag_rel), 1) + 0.05)
        plt.show()


class Characteristics(metaclass=ABCMeta):
    def __init__(self, extend: float=0.2, grid_pnt_num: int=50):
        self.extend = extend
        self.grid_pnt_num = grid_pnt_num
        self.G_grid = None
        self._G_interp = None
        self.eta_c_grid = None
        self._eta_c_interp = None
        self.branches: typing.List[FrequencyBranch] = None

    def get_G_grid(self, extend: float = 0.2, pnt_num: int = 50):
        frequency = []
        G = []
        pi_c = []
        for branch in self.branches:
            pi_c_int = max(branch.pi_c_stag_rel) - min(branch.pi_c_stag_rel)
            pi_c_extend = pi_c_int * extend
            pi_c_min = min(branch.pi_c_stag_rel) - pi_c_extend
            pi_c_max = max(branch.pi_c_stag_rel) + pi_c_extend
            pi_c_arr, G_arr = branch.get_G_spline_points(pi_c_min, pi_c_max, pnt_num)
            G.append(list(G_arr))
            pi_c.append(list(pi_c_arr))
            frequency.append([branch.freq_norm_rel for _ in pi_c_arr])
        return frequency, pi_c, G

    def get_eta_c_grid(self, extend: float = 0.2, pnt_num: int = 50):
        frequency = []
        G = []
        eta_c = []
        for branch in self.branches:
            G_int = max(branch.G_norm_rel) - min(branch.G_norm_rel)
            G_extend = G_int * extend
            G_min = min(branch.G_norm_rel) - G_extend
            G_max = max(branch.G_norm_rel) + G_extend
            G_arr, eta_c_arr = branch.get_eta_c_stag_spline_points(G_min, G_max, pnt_num)
            G.append(list(G_arr))
            eta_c.append(list(eta_c_arr))
            frequency.append([branch.freq_norm_rel for _ in eta_c_arr])
        return frequency, G, eta_c,

    def get_eta_c_stag_rel(self, freq_norm_rel, G_norm_rel):
        assert self._eta_c_interp is not None, 'Grid must be computed.'
        return self._eta_c_interp(freq_norm_rel, G_norm_rel)

    def get_G_norm_rel(self, freq_norm_rel, pi_c_stag_rel):
        assert self._G_interp is not None, 'Grid must be computed.'
        return self._G_interp(freq_norm_rel, pi_c_stag_rel)

    def compute(self):
        for branch in self.branches:
            branch.compute()
        self.G_grid = self.get_G_grid(self.extend, self.grid_pnt_num)
        self.eta_c_grid = self.get_eta_c_grid(self.extend, self.grid_pnt_num)
        self._G_interp = Interp2D(self.G_grid[0], self.G_grid[1], self.G_grid[2])
        self._eta_c_interp = Interp2D(self.eta_c_grid[0], self.eta_c_grid[1], self.eta_c_grid[2])

    @classmethod
    def combine_branches(cls, branches: typing.List[typing.List[float]]) -> list:
        res = []
        for branch in branches:
            res += branch
        return res

    def plot_grid_3d(self, figsize=(8, 6)):
        fig_G = plt.figure(figsize=figsize)
        ax_G = Axes3D(fig_G)
        ax_G.scatter(
            self.combine_branches(self.G_grid[0]),
            self.combine_branches(self.G_grid[1]),
            self.combine_branches(self.G_grid[2]),
            s=10
        )
        ax_G.set_xlabel(r'$\bar{n}_{пр}$', fontsize=12)
        ax_G.set_ylabel(r'$\bar{\pi}_{к}^*$', fontsize=12)
        ax_G.set_zlabel(r'$\bar{G}_{пр}$', fontsize=12)
        plt.show()

        fig_eta = plt.figure(figsize=figsize)
        ax_eta = Axes3D(fig_eta)
        ax_eta.scatter(
            self.combine_branches(self.eta_c_grid[0]),
            self.combine_branches(self.eta_c_grid[1]),
            self.combine_branches(self.eta_c_grid[2]),
            s=10
        )
        ax_eta.set_xlabel(r'$\bar{n}_{пр}$', fontsize=12)
        ax_eta.set_ylabel(r'$\bar{\pi}_{к}^*$', fontsize=12)
        ax_eta.set_zlabel(r'$\bar{\eta}_{к}$', fontsize=12)
        plt.show()

    def plot_grid_2d(self, figsize=(7, 10)):
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.scatter(
            self.combine_branches(self.G_grid[2]),
            self.combine_branches(self.G_grid[1]),
            s=8, color='blue', label='Spline grid'
        )
        for n, branch in enumerate(self.branches):
            if n == 0:
                plt.plot(
                    branch.G_norm_rel, branch.pi_c_stag_rel, color='red', ls='--', lw=1, marker='s', ms=3, label='Init'
                )
            else:
                plt.plot(
                    branch.G_norm_rel, branch.pi_c_stag_rel, color='red', ls='--', lw=1, marker='s', ms=3
                )
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\pi}_{к}^*$', fontsize=10)
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.scatter(
            self.combine_branches(self.eta_c_grid[1]),
            self.combine_branches(self.eta_c_grid[2]),
            s=8, color='blue', label='Spline grid'
        )
        for n, branch in enumerate(self.branches):
            if n == 0:
                plt.plot(
                    branch.G_norm_rel, branch.eta_c_stag_rel, color='red', ls='--', lw=1, marker='s', ms=3, label='Init'
                )
            else:
                plt.plot(
                    branch.G_norm_rel, branch.eta_c_stag_rel, color='red', ls='--', lw=1, marker='s', ms=3
                )
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\eta}_{к}$', fontsize=10)
        plt.ylim(
            round(min(self.combine_branches(self.eta_c_grid[2])), 1) + 0.2,
            round(max(self.combine_branches(self.eta_c_grid[2])), 1) + 0.05
        )
        plt.grid()
        plt.legend()
        plt.show()

    def plot_modes_line(self, pi_c_stag_rel_arr, G_norm_rel_arr, eta_c_stag_rel_arr, figsize=(7, 6), fname_base=None):
        plt.figure(figsize=figsize)
        for n, branch in enumerate(self.branches):
            if n == 0:
                plt.plot(
                    branch.G_norm_rel, branch.pi_c_stag_rel, color='red', ls='-', lw=1, marker='s', ms=3, label='1'
                )
            else:
                plt.plot(
                    branch.G_norm_rel, branch.pi_c_stag_rel, color='red', ls='-', lw=1, marker='s', ms=3
                )
        plt.plot(G_norm_rel_arr, pi_c_stag_rel_arr, color='black', lw=1.5, marker='o', ms=3, label='2')
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\pi}_{к}^*$', fontsize=10)
        plt.grid()
        plt.legend(fontsize=12)
        if fname_base:
            plt.savefig(fname_base + 'G_pi_c_axis.png')
        plt.show()

        plt.figure(figsize=figsize)
        for n, branch in enumerate(self.branches):
            if n == 0:
                plt.plot(
                    branch.G_norm_rel, branch.eta_c_stag_rel, color='red', ls='-', lw=1, marker='s', ms=3, label='1'
                )
            else:
                plt.plot(
                    branch.G_norm_rel, branch.eta_c_stag_rel, color='red', ls='-', lw=1, marker='s', ms=3
                )
        plt.plot(G_norm_rel_arr, eta_c_stag_rel_arr, color='black', lw=1.5, marker='o', ms=3, label='2')
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\eta}_{к}$', fontsize=10)
        plt.grid()
        plt.legend(fontsize=12)
        if fname_base:
            plt.savefig(fname_base + 'G_pi_c_axis.png')
        plt.show()

    def plot_branches_2d(self, frequency_int=(0.65, 1.15), branch_num: int = 10, figsize=(7, 10), pnt_num=1000):
        frequencies = np.linspace(frequency_int[0], frequency_int[1], branch_num)

        plt.figure(figsize=figsize)
        plt.subplot(211)
        for branch in self.branches:
            plt.scatter(branch.G_norm_rel, branch.pi_c_stag_rel, c='blue', s=10)
        pi_c_arr = np.linspace(
            min(self.combine_branches(self.G_grid[1])),
            max(self.combine_branches(self.G_grid[1])), pnt_num
        )
        for frequency in frequencies:
            G_pi_arr = [self.get_G_norm_rel(frequency, pi_c) for pi_c in pi_c_arr]
            plt.plot(G_pi_arr, pi_c_arr, color='red', lw=1)
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\pi}_{к}^*$', fontsize=10)
        plt.xlim(
            round(min(self.combine_branches(self.G_grid[2])), 1) - 0.05,
            round(max(self.combine_branches(self.G_grid[2])), 1) + 0.05
        )
        plt.ylim(
            round(min(self.combine_branches(self.G_grid[1])), 1) - 0.05,
            round(max(self.combine_branches(self.G_grid[1])), 1) + 0.05
        )
        plt.grid()

        plt.subplot(212)
        for branch in self.branches:
            plt.scatter(branch.G_norm_rel, branch.eta_c_stag_rel, c='blue', s=10)
        G_eta_arr = np.linspace(
            min(self.combine_branches(self.eta_c_grid[1])),
            max(self.combine_branches(self.eta_c_grid[1])), pnt_num
        )
        for frequency in frequencies:
            eta_c_arr = [self.get_eta_c_stag_rel(frequency, G) for G in G_eta_arr]
            plt.plot(G_eta_arr, eta_c_arr, color='red', lw=1)
        plt.xlabel(r'$\bar{G}_{пр}$', fontsize=10)
        plt.ylabel(r'$\bar{\eta}_{к}$', fontsize=10)
        plt.xlim(
            round(min(self.combine_branches(self.eta_c_grid[1])), 1) - 0.05,
            round(max(self.combine_branches(self.eta_c_grid[1])), 1) + 0.05
        )
        plt.ylim(
            round(min(self.combine_branches(self.eta_c_grid[2])), 1) + 0.6,
            round(max(self.combine_branches(self.eta_c_grid[2])), 1) + 0.05
        )
        plt.grid()
        plt.show()