import math
import random
from typing import List, Tuple
import numpy as np


class Wolff:
    
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
        self.delta = lambda x,y: 1 if x==y else 0   # kcronecker delta function.

    def wolff(self) -> None:
        # Select one vertex at random.
        x = random.randint(0, self.Nx-1)
        y = random.randint(0, self.Ny-1)

        # Add to seed.
        seed = []
        cluster = []
        seed.append((x, y))
        cluster.append((x, y))

        # Calculate Padd.
        Padd = 1.0 - math.exp(-2.0 * self.J / self.T)

        # Loop run until seed is empty.
        while seed:
            # Choice one vertex from seed.
            idx = random.randint(0, len(seed)-1)
            coord = seed.pop(idx)

            # Loop run nearest-neighbor spins.
            for nbr in self.nbrs(*coord):
                # If they are pointing in the same direction as the seed spin,
                # add them to the cluster with probability Padd = 1 - exp(-2βJ/T)
                if self.spins[coord] == self.spins[nbr] and nbr not in cluster and random.random() < Padd:
                    seed.append(nbr)
                    cluster.append(nbr)

        # update cluster spins.
        s = random.randint(0, self.q-1)
        for coord in cluster:
            self.spins[coord] = s

    def mcstep(self) -> None:
        self.wolff()

if __name__ == '__main__':
    q = 2
    Nx = 16
    Ny = 16
    J = 1.0
    T = 1.0
    model = Wolff(q, Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(1000):
        model.mcstep()
        # print(f'M = {model.spins.sum()}')
        plt.imshow(model.spins)
        plt.pause(0.001)
        plt.show()