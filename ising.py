import numpy as np
import enum
from matplotlib import pyplot as plt
import dataclasses
from typing import List
import json

plt.rcParams.update({"font.size": 25})


class InitialState(enum.Enum):
    RANDOM = 0
    UP = 1
    DOWN = 2


class Event:
    pass


class FlipSpin(Event):
    def __init__(self, new_value):
        self.new_value = new_value


class Method(enum.Enum):
    METROPOLIS = 0
    GLAUBER = 1


@dataclasses.dataclass
class Info:
    magnetization: List[float]

    def to_json(self, output_file):
        with open(output_file, "w") as f:
            f.write(json.dumps(dataclasses.asdict(self), indent=4))


class Simulation:
    def __init__(
        self,
        length,
        temperature,
        magnetic_field,
        magnetization_coefficient,
        dim,
        initial_state=InitialState.RANDOM,
        method=Method.METROPOLIS,
    ):
        self.length = length
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.magnetization_coefficient = magnetization_coefficient
        self.dim = dim
        self.lattice = self._get_initial_lattice(dim, length, initial_state)
        self.method = method
        self.info = self._get_init_info()

    def run(self, number_of_steps):
        for _ in range(number_of_steps):
            self._run_step()

    def show_lattice(self, output_file=None):
        if self.dim == 1:
            plt.imshow([self.lattice] * 100, cmap="gray")
            if output_file:
                plt.savefig(output_file, bbox_inches="tight")
                plt.clf()
            else:
                plt.show()

    def save_results(self, output_file):
        # save self.info as json
        self.info.to_json(output_file)

    def plot_magnetization(self, output_file=None):
        plt.plot(range(len(self.info.magnetization)), self.info.magnetization)
        plt.xlabel("Step")
        plt.ylabel("Magnetization")
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
            plt.clf()
        else:
            plt.show()

    def _get_init_info(self):
        return Info(magnetization=[np.sum(self.lattice)])

    def _save_step(self, events):
        spin_flip_events = [event for event in events if isinstance(event, FlipSpin)]
        if len(spin_flip_events) == 0:
            self.info.magnetization.append(self.info.magnetization[-1])
        else:
            self.info.magnetization.append(
                self.info.magnetization[-1]
                + 2 * np.sum([spin_flip.new_value for spin_flip in spin_flip_events])
            )

    def _run_step(self):
        if self.method == Method.METROPOLIS:
            events = self._run_metropolis_step()
        elif self.method == Method.GLAUBER:
            events = self._run_glauber_step()
        else:
            raise ValueError("Invalid method")

        self._save_step(events)

    def _run_metropolis_step(self):
        events = []
        flip_spin = np.random.randint(0, self.length, size=self.dim)
        delta_energy = self._get_delta_energy(flip_spin)
        if delta_energy < 0:
            self.lattice[flip_spin] *= -1
            events.append(FlipSpin(self.lattice[flip_spin]))
        else:
            if np.random.random() < np.exp(-delta_energy / self.temperature):
                self.lattice[flip_spin] *= -1
                events.append(FlipSpin(self.lattice[flip_spin]))

        return events

    def _get_delta_energy(self, flip_spin):
        current_energy = self._get_energy(self.lattice)
        self.lattice[flip_spin] *= -1
        new_energy = self._get_energy(self.lattice)
        self.lattice[flip_spin] *= -1
        return new_energy - current_energy

    def _get_energy(self, spins):
        return -self.magnetic_field * np.sum(
            spins
        ) - self.magnetization_coefficient * np.sum(spins * np.roll(spins, 1, axis=0))

    def _run_glauber_step(self):
        return []

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
        magnetic_field=100,
        magnetization_coefficient=1,
        dim=1,
        initial_state=InitialState.DOWN,
        method=Method.METROPOLIS,
    )
    simulation.show_lattice("output/ising_test/lattice_before.png")
    simulation.run(1000)
    simulation.show_lattice("output/ising_test/lattice_after.png")
    simulation.save_results("output/ising_test/data.json")
    simulation.plot_magnetization("output/ising_test/magnetization.png")


if __name__ == "__main__":
    main()
