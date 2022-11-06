import numpy as np
import matplotlib.pyplot as plt
from samplers.metropolis import Metropolis
from samplers.swendsen_wang import SwendsenWang
from samplers.wolff import Wolff

if __name__  == '__main__':
    # config
    q = 3
    Nx = 64
    Ny = 64
    J = 1.0
    T = 2.0
    Step = 1000

    # Create sampler
    # model = Metropolis(q, Nx, Ny, J, T)
    model = SwendsenWang(q, Nx, Ny, J, T)
    # model = Wolff(q, Nx, Ny, J, T)

    try:
        plt.ion()
        fig = plt.figure()
        for i in range(Step):
            model.mcstep()
            plt.imshow(model.spins)
            plt.pause(0.001)
            plt.show()

    except KeyboardInterrupt:
        plt.ioff()
        plt.close(fig)