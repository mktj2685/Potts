import math
import random
from typing import List, Tuple
import numpy as np
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils.unionfind import UnionFind


class SwendsenWang:
    
    def __init__(
        self,
        q: int,
        Nx: int,
        Ny: int,
        J: float,
        T: float
    ) -> None:
        # Set parameters.
        self.q = q
        self.Nx = Nx
        self.Ny = Ny
        self.J = J
        self.T = T
        self.spins = np.random.randint(0, q, (Nx, Ny))
        self.nbrs = lambda x, y: [
            (x, (y+1)%self.Ny),     # above
            ((x+1)%self.Nx, y)      # right
        ]

    def swendsen_wang(self) -> None:
        # Create Union-Find
        uf = UnionFind(self.Nx*self.Ny)

        # Calculate Padd.
        Padd = 1.0 - math.exp(-2.0 * self.J / self.T)

        # Loop run vertices belongs bulk.
        for x in range(self.Nx):
            for y in range(self.Ny):
                for x_, y_ in self.nbrs(x, y):
                    if self.spins[x,y] == self.spins[x_, y_] and random.random() < Padd:
                        i = y * self.Nx + x
                        j = y_ * self.Nx + x_
                        uf.union(i, j)

        # update luster spins.
        for root in uf.roots():
            s = random.randint(0, self.q-1)
            for idx in uf.members(root):
                x = idx % self.Nx
                y = idx // self.Nx
                self.spins[x, y] = s

    def mcstep(self) -> None:
        self.swendsen_wang()

if __name__ == '__main__':
    q = 2
    Nx = 16
    Ny = 16
    J = 1.0
    T = 3.0
    model = SwendsenWang(q, Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(1000):
        model.mcstep()
        # print(f'M = {model.spins.sum()}')
        plt.imshow(model.spins)
        plt.pause(0.001)
        plt.show()