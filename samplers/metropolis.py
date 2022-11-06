import math
import random
import numpy as np

class Metropolis:

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
        self.e = self.energy()                      # energy of current configuration.

    def energy(self) -> float:
        e = 0.0
        for x in range(self.Nx):
            for y in range(self.Ny):
                for nbr in self.nbrs(x,y):
                    e += -self.J * self.delta(self.spins[x,y], self.spins[nbr])
        return e

    def metropolis(self) -> None:
        # Select one vertex at random.
        x = random.randint(0, self.Nx-1)
        y = random.randint(0, self.Ny-1)

        # Flip spin
        s = self.spins[x, y]
        self.spins[x, y] = random.randint(0, self.q-1)
        
        # Calculate the energy after spin flipped.
        e = self.energy()

        # Calculate the energy difference.
        de = e - self.e

        # accept
        if de < 0 or random.random() < math.exp(-de / self.T):
            self.e = e

        # reject
        else:
            self.spins[x, y] = s

    def mcstep(self) -> None:
        self.metropolis()

if __name__ == '__main__':
    q = 2
    Nx = 16
    Ny = 16
    J = 1.0
    T = 2.5

    model = Metropolis(q, Nx, Ny, J, T)

    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(500000):
        model.mcstep()
        print(f'M = {model.spins.sum()}')