import numpy as np
import enum


class InitialState(enum.Enum):
    RANDOM = 0
    UP = 1
    DOWN = 2


class Method(enum.Enum):
    METROPOLIS = 0
    GLAUBER = 1


class Simulation:
    def __init__(
        self,
        length,
        temperature,
        magnetic_field,
        dim,
        initial_state=InitialState.RANDOM,
        method=Method.METROPOLIS,
    ):
        self.length = length
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.dim = dim
        self.lattice = self._get_initial_lattice(dim, length, initial_state)
        self.method = method

    def run(self, number_of_steps):
        pass

    def run_step(self):
        pass

    @staticmethod
    def _get_initial_lattice(dim, length, initial_state):
        if initial_state == InitialState.RANDOM:
            return np.random.choice([-1, 1], size=(length,) * dim)
        elif initial_state == InitialState.UP:
            return np.ones((length,) * dim)
        elif initial_state == InitialState.DOWN:
            return -np.ones((length,) * dim)
        else:
            raise ValueError("Invalid initial state")


def main():
    simulation = Simulation(
        length=512,
        temperature=1,
        magnetic_field=1,
        dim=1,
        initial_state=InitialState.RANDOM,
        method=Method.METROPOLIS,
    )
