import numpy as np
import enum
from matplotlib import pyplot as plt
import dataclasses
from typing import List
import json
import copy
import os

np.random.seed(42)
plt.rcParams.update({"font.size": 25})


class InitialState(enum.Enum):
    RANDOM = 0
    UP = 1
    DOWN = 2


class FlipDynamics(enum.Enum):
    SINGLE = 0
    COMPLETE = 1


class Event:
    pass


class FlipSpin(Event):
    def __init__(self, spin, new_value):
        self.spin = spin
        self.new_value = new_value

    def accept(self, simulation):
        simulation.lattice[self.spin] = self.new_value


class NewRandomLattice(Event):
    def __init__(self, new_lattice):
        self.new_lattice = new_lattice

    def accept(self, simulation):
        simulation.lattice = self.new_lattice


class Method(enum.Enum):
    METROPOLIS = 0
    GLAUBER = 1


@dataclasses.dataclass
class Info:
    magnetization: List[int]
    number_of_spins: int

    @property
    def mean_magnetization(self):
        return [int(m) / self.number_of_spins for m in self.magnetization]

    def to_json(self, output_file):
        with open(output_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "magnetization": [int(m) for m in self.magnetization],
                        "mean_magnetization": self.mean_magnetization,
                    },
                    indent=4,
                )
            )


class Simulation:
    def __init__(
        self,
        name,
        length,
        temperature,
        magnetic_field,
        magnetization_coefficient,
        dim,
        initial_state=InitialState.RANDOM,
        method=Method.METROPOLIS,
        flip_dynamics=FlipDynamics.SINGLE,
        initial_lattice=None,
    ):
        self.name = name
        self.length = length
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.magnetization_coefficient = magnetization_coefficient
        self.dim = dim
        if initial_lattice is not None:
            self.lattice = copy.deepcopy(initial_lattice)
        else:
            self.lattice = self._get_initial_lattice(dim, length, initial_state)

        self.method = method
        self.flip_dynamics = flip_dynamics
        self.number_of_spins = length**dim
        self.info = self._get_init_info()

    def run(self, number_of_steps):
        for _ in range(number_of_steps):
            self._run_step()

    def show_lattice(self, output_file=None):
        if self.dim == 1:
            # black for -1, white for 1
            plt.imshow([self.lattice] * 100, cmap="gray", vmin=-1, vmax=1)

            plt.yticks([])
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
        return Info(
            number_of_spins=self.number_of_spins, magnetization=[np.sum(self.lattice)]
        )

    def _save_step(self, events):
        spin_flip_events = [event for event in events if isinstance(event, FlipSpin)]
        new_random_lattice_events = [
            event for event in events if isinstance(event, NewRandomLattice)
        ]
        if len(spin_flip_events) != 0 and len(new_random_lattice_events) != 0:
            raise ValueError("Invalid events")

        elif len(spin_flip_events) == 0 and len(new_random_lattice_events) == 0:
            self.info.magnetization.append(self.info.magnetization[-1])
        elif len(spin_flip_events) == 0:
            self.info.magnetization.append(np.sum(self.lattice))
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

    def _get_potential_events(self):
        if self.flip_dynamics == FlipDynamics.SINGLE:
            flip_spin = np.random.randint(0, self.length, size=self.dim)
            return [FlipSpin(flip_spin, -1 * self.lattice[flip_spin])]
        elif self.flip_dynamics == FlipDynamics.COMPLETE:
            new_random_lattice = np.random.choice(
                [-1, 1], size=(self.length,) * self.dim
            )
            return [NewRandomLattice(new_random_lattice)]
        else:
            raise ValueError("Invalid flip dynamics")

    def _run_metropolis_step(self):
        events = []
        potential_events = self._get_potential_events()
        for potential_event in potential_events:
            delta_energy = self._get_delta_energy(potential_event)
            if delta_energy < 0:
                potential_event.accept(self)
                events.append(potential_event)
            else:
                if np.random.random() < np.exp(-delta_energy / self.temperature):
                    potential_event.accept(self)
                    events.append(potential_event)

            return events

    def _get_delta_energy(self, event):
        current_energy = self._get_energy(self.lattice)
        if isinstance(event, FlipSpin):
            self.lattice[event.spin] = event.new_value
            energy_diff = self._get_energy(self.lattice) - current_energy
            self.lattice[event.spin] = event.new_value * -1
            return energy_diff

        elif isinstance(event, NewRandomLattice):
            return self._get_energy(event.new_lattice) - current_energy

    def _get_energy(self, spins):
        return -self.magnetic_field * np.sum(
            spins
        ) - self.magnetization_coefficient * np.sum(spins * np.roll(spins, 1, axis=0))

    def _run_glauber_step(self):
        events = []
        potential_events = self._get_potential_events()
        for potential_event in potential_events:
            delta_energy = self._get_delta_energy(potential_event)
            if np.random.random() < 1 / (1 + np.exp(delta_energy / self.temperature)):
                potential_event.accept(self)
                events.append(potential_event)

        return events

    @staticmethod
    def _get_initial_lattice(dim, length, initial_state):
        if initial_state == InitialState.UP:
            return np.int8(np.ones((length,) * dim))
        elif initial_state == InitialState.DOWN:
            return -np.int8(np.ones((length,) * dim))
        elif initial_state == InitialState.RANDOM:
            return np.int8(np.random.choice([-1, 1], size=(length,) * dim))
        else:
            raise ValueError("Invalid initial state")


def run_thermalization_period(method, flip_dynamics, output_dir):
    # create outpuut dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_random_initial = Simulation(
        name="random_initial",
        length=512,
        temperature=1,
        magnetic_field=1,
        magnetization_coefficient=1,
        dim=1,
        initial_state=InitialState.RANDOM,
        method=method,
        flip_dynamics=flip_dynamics,
    )
    simulation_all_up_initial = Simulation(
        name="all_up_initial",
        length=512,
        temperature=1,
        magnetic_field=1,
        magnetization_coefficient=1,
        dim=1,
        initial_state=InitialState.UP,
        method=method,
        flip_dynamics=flip_dynamics,
    )
    compare_mean_magnetization(
        [simulation_random_initial, simulation_all_up_initial],
        output_dir,
    )


def method_comparison():
    metropolis_simulation = Simulation(
        name="metropolis",
        length=512,
        temperature=1,
        magnetic_field=1,
        magnetization_coefficient=1,
        dim=1,
        initial_state=InitialState.RANDOM,
        method=Method.METROPOLIS,
        flip_dynamics=FlipDynamics.SINGLE,
    )
    glauber_simulation = Simulation(
        name="glauber",
        length=512,
        temperature=1,
        magnetic_field=1,
        magnetization_coefficient=1,
        dim=1,
        initial_lattice=metropolis_simulation.lattice,
        method=Method.GLAUBER,
        flip_dynamics=FlipDynamics.SINGLE,
    )
    compare_mean_magnetization(
        [metropolis_simulation, glauber_simulation],
        "output/method_comparison",
    )


def compare_mean_magnetization(simulations, output_dir):
    for simulation in simulations:
        simulation.show_lattice(f"{output_dir}/lattice_{simulation.name}_00000.png")
        simulation.run(1000)
        simulation.show_lattice(f"{output_dir}/lattice_{simulation.name}_01000.png")
        simulation.run(9000)
        simulation.show_lattice(f"{output_dir}/lattice_{simulation.name}_10000.png")
        simulation.save_results(f"{output_dir}/results_{simulation.name}_10000.json")

    for simulation in simulations:
        plt.plot(
            range(len(simulation.info.mean_magnetization)),
            simulation.info.mean_magnetization,
            label=simulation.name.capitalize().replace("_", " "),
        )

    plt.xlabel("Step")
    plt.ylabel("Mean magnetization")
    plt.legend()
    plt.savefig(f"{output_dir}/mean_magnetization.png", bbox_inches="tight")
    plt.clf()


def run_thermalization_periods():
    run_thermalization_period(
        Method.METROPOLIS,
        FlipDynamics.SINGLE,
        "output/thermalization/metropolis/single",
    )
    run_thermalization_period(
        Method.METROPOLIS,
        FlipDynamics.COMPLETE,
        "output/thermalization/metropolis/complete",
    )
    run_thermalization_period(
        Method.GLAUBER,
        FlipDynamics.SINGLE,
        "output/thermalization/glauber/single",
    )
    run_thermalization_period(
        Method.GLAUBER,
        FlipDynamics.COMPLETE,
        "output/thermalization/glauber/complete",
    )


if __name__ == "__main__":
    # run_thermalization_periods()
    method_comparison()
