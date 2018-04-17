import typing


class Interp2D:
    def __init__(self, x_branches: typing.List[typing.List[float]],
                 y_branches: typing.List[typing.List[float]],
                 z_branches: typing.List[typing.List[float]]):
        """
        x_branches, y_branches, z_branches - списки, которые содержат списки значений координат для каждой ветки с
        постоянным значением x.
        Ветви должны находиться в порядке возрастания x. В каждой ветви значения должны быть расположены в порядке
        возрастания y.
        """
        self.x_branches = x_branches
        self.y_branches = y_branches
        self.z_branches = z_branches

    def check_dimension(self):
        for x_br, y_br, z_br in zip(self.x_branches, self.y_branches, self.z_branches):
            if not (len(x_br) == len(y_br) == len(z_br)):
                return False
        return True

    def check_x_values(self):
        for x_br in self.x_branches:
            if x_br.count(x_br[0]) != len(x_br):
                return False
        return True

    def check_branches_arrangement(self):
        for i in range(len(self.x_branches) - 1):
            if self.x_branches[i][0] >= self.x_branches[i + 1][0]:
                return False
        return True

    def check_points_arrangement(self):
        for y_br in self.y_branches:
            for i in range(len(y_br) - 1):
                if y_br[i + 1] <= y_br[i]:
                    return False
        return True

    def _check(self):
        if not self.check_dimension():
            raise Exception('The length of value lists for x, y and z must match for all branches.')
        if not self.check_x_values():
            raise Exception('Every branch must contain values of coordinates for points with the same value of x.')
        if not self.check_points_arrangement():
            raise Exception('Values in every branch must be arranged in ascending order of y.')
        if not self.check_branches_arrangement():
            raise Exception('Branches must be arrange in ascending order of z.')

    @classmethod
    def get_value_from_branch(cls, y_int, y_br: typing.List[float],
                              z_br: typing.List[float]):
        z_int = None
        if y_int >= y_br[len(y_br) - 1]:
            int_factor = (z_br[len(z_br) - 1] - z_br[len(z_br) - 2]) / (y_br[len(y_br) - 1] - y_br[len(y_br) - 2])
            z_int = z_br[len(z_br) - 1] + (y_int - y_br[len(y_br) - 1]) * int_factor
        elif y_int < y_br[0]:
            int_factor = (z_br[1] - z_br[0]) / (y_br[1] - y_br[0])
            z_int = z_br[0] + int_factor * (y_int - y_br[0])
        else:
            for i in range(len(y_br) - 1):
                if y_br[i] <= y_int < y_br[i + 1]:
                    int_factor = (z_br[i + 1] - z_br[i]) / (y_br[i + 1] - y_br[i])
                    z_int = z_br[i] + (y_int - y_br[i]) * int_factor
        return z_int

    @classmethod
    def get_intermediate_branch(cls, x_int: float, x_br1: typing.List[float], y_br1: typing.List[float],
                                z_br1: typing.List[float], x_br2: typing.List[float],
                                y_br2: typing.List[float], z_br2: typing.List[float]):
        x_res = []
        y_res = []
        z_res = []
        for i in range(len(x_br1)):
            x = x_int
            y = y_br1[i] + (y_br2[i] - y_br1[i]) / (x_br2[i] - x_br1[i]) * (x_int - x_br1[i])
            z = z_br1[i] + (z_br2[i] - z_br1[i]) / (x_br2[i] - x_br1[i]) * (x_int - x_br1[i])
            x_res.append(x)
            y_res.append(y)
            z_res.append(z)
        return x_res, y_res, z_res

    def __call__(self, *args, **kwargs):
        x_int = args[0]
        y_int = args[1]
        z_int = None
        self._check()
        for i in range(len(self.x_branches) - 1):
            if self.x_branches[i][0] <= x_int < self.x_branches[i + 1][0]:
                x_interm, y_interm, z_interm = self.get_intermediate_branch(
                    x_int, self.x_branches[i], self.y_branches[i], self.z_branches[i],
                    self.x_branches[i + 1], self.y_branches[i + 1], self.z_branches[i + 1]
                )
                z_int = self.get_value_from_branch(y_int, y_interm, z_interm)
            elif self.x_branches[len(self.x_branches) - 1][0] <= x_int:
                x_interm, y_interm, z_interm = self.get_intermediate_branch(
                    x_int, self.x_branches[len(self.x_branches) - 2], self.y_branches[len(self.x_branches) - 2],
                    self.z_branches[len(self.x_branches) - 2],
                    self.x_branches[len(self.x_branches) - 1], self.y_branches[len(self.x_branches) - 1],
                    self.z_branches[len(self.x_branches) - 1]
                )
                z_int = self.get_value_from_branch(y_int, y_interm, z_interm)
            elif self.x_branches[0][0] > x_int:
                x_interm, y_interm, z_interm = self.get_intermediate_branch(
                    x_int, self.x_branches[0], self.y_branches[0], self.z_branches[0],
                    self.x_branches[1], self.y_branches[1], self.z_branches[1]
                )
                z_int = self.get_value_from_branch(y_int, y_interm, z_interm)
        return z_int
