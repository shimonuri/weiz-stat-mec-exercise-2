import numpy as np
import math
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
    COMPLETE_RANDOM_STEP = 1


class Event:
    pass


class FlipSpin(Event):
    def __init__(self, spin, new_value):
        self.spin = spin
        self.new_value = new_value
        self.old_value = -1 * new_value

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
    magnetization: List[float]
    energy: List[float]
    number_of_spins: int
    temperature: float
    correlation: List[np.ndarray] = dataclasses.field(default_factory=list)

    @property
    def mean_magnetization(self):
        return [
            np.mean(self.magnetization[i : i + self.number_of_spins])
            / self.number_of_spins
            for i in range(0, len(self.magnetization), self.number_of_spins)
        ]

    @property
    def mean_magnetization_at_end_of_sweep(self):
        return [
            self.magnetization[i] / self.number_of_spins
            for i in range(0, len(self.magnetization), self.number_of_spins)
        ]

    @property
    def mean_energies(self):
        return [
            np.mean(self.energy[i : i + self.number_of_spins])
            for i in range(0, len(self.energy), self.number_of_spins)
        ]

    @property
    def mean_energies_at_end_of_sweep(self):
        return [
            self.energy[i]
            for i in range(0, len(self.energy), self.number_of_spins)
        ]

    @property
    def mean_correlation(self):
        # mean between all the correlations
        return np.mean(self.correlation, axis=0)

    @property
    def mean_energy(self):
        return np.mean(self.mean_energies)

    @property
    def mean_energy_at_end_of_sweep(self):
        return np.mean(self.mean_energies_at_end_of_sweep)

    @property
    def specific_heat(self):
        return np.var(self.mean_energies) / (self.temperature**2)

    @property
    def specific_heat_at_end_of_sweep(self):
        return np.var(self.mean_energies_at_end_of_sweep) / (self.temperature**2)

    @property
    def mean_magnetization_per_step(self):
        return [m / self.number_of_spins for m in self.magnetization]

    def append_correlation(self, lattice, max_size):
        self.correlation.append(
            np.array([np.mean(lattice * np.roll(lattice, x)) for x in range(max_size)])
        )

    def get_number_of_steps_to_mean_magnetization(self, mean_magnetization):
        # return the number of steps to reach the given magnetization
        for i, m in enumerate(self.mean_magnetization_per_step):
            if m >= mean_magnetization:
                return i

        raise ValueError("Magnetization not reached")

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
        self.current_energy = self._get_energy(self.lattice)
        self.step_made = 0

    def run(self, number_of_sweeps, should_save=True):
        for _ in range(number_of_sweeps * self.number_of_spins):
            events = self._run_step()
            if should_save:
                self.step_made += 1
                self._save_step(events)

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

    def plot_mean_magnetization(self, output_file=None):
        plt.plot(range(len(self.info.mean_magnetization)), self.info.mean_magnetization)
        plt.xlabel("Sweep")
        plt.ylabel("Magnetization")
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
            plt.clf()
        else:
            plt.show()

    def plot_mean_energy(self, output_file=None):
        mean_energy = self.info.mean_energies()
        plt.plot(range(len(mean_energy)), mean_energy)
        plt.xlabel("Sweep")
        plt.ylabel("Energy")
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
            plt.clf()
        else:
            plt.show()

    def _get_init_info(self):
        return Info(
            number_of_spins=self.number_of_spins,
            energy=[self._get_energy(self.lattice)],
            magnetization=[np.sum(self.lattice)],
            temperature=self.temperature,
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
            self.info.energy.append(self.current_energy)
        elif len(spin_flip_events) == 0:
            self.info.magnetization.append(np.sum(self.lattice))
            self.info.energy.append(self.current_energy)
        else:
            self.info.magnetization.append(
                self.info.magnetization[-1]
                + 2 * np.sum([spin_flip.new_value for spin_flip in spin_flip_events])
            )
            self.info.energy.append(self.current_energy)

        if self.step_made % self.number_of_spins == 0:
            self.info.append_correlation(self.lattice, self.length)

    def _run_step(self):
        if self.method == Method.METROPOLIS:
            events = self._run_metropolis_step()
        elif self.method == Method.GLAUBER:
            events = self._run_glauber_step()
        else:
            raise ValueError("Invalid method")

        return events

    def _get_potential_events(self):
        if self.flip_dynamics == FlipDynamics.SINGLE:
            flip_spin = np.random.randint(0, self.length, size=self.dim)
            return [FlipSpin(flip_spin, -1 * self.lattice[flip_spin])]
        elif self.flip_dynamics == FlipDynamics.COMPLETE_RANDOM_STEP:
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
                self._accept_event(potential_event, delta_energy)
                events.append(potential_event)
            else:
                if np.random.random() < np.exp(-delta_energy / self.temperature):
                    self._accept_event(potential_event, delta_energy)
                    events.append(potential_event)

            return events

    def _get_delta_energy(self, event):
        if isinstance(event, FlipSpin):
            return self._get_flip_spin_energy_diff(event)

        elif isinstance(event, NewRandomLattice):
            return self._get_energy(event.new_lattice) - self.current_energy

    def _get_flip_spin_energy_diff(self, spin_flip):
        next_spin = self.lattice[(spin_flip.spin + 1) % self.length]
        prev_spin = self.lattice[spin_flip.spin - 1]
        energy_diff = 2 * (
            self.magnetic_field * spin_flip.old_value
            + self.magnetization_coefficient
            * spin_flip.old_value
            * (next_spin + prev_spin)
        )
        return float(energy_diff)

    def _get_energy(self, spins):
        return -self.magnetic_field * np.sum(
            spins
        ) - self.magnetization_coefficient * np.sum(spins * np.roll(spins, 1, axis=0))

    def _accept_event(self, event, energy_diff):
        event.accept(self)
        self.current_energy += energy_diff

    def _run_glauber_step(self):
        events = []
        potential_events = self._get_potential_events()
        for potential_event in potential_events:
            delta_energy = self._get_delta_energy(potential_event)
            if np.random.random() < 1 / (1 + np.exp(delta_energy / self.temperature)):
                self._accept_event(potential_event, delta_energy)
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


def run_thermalization_period(method, flip_dynamics, output_dir=None):
    # create outpuut dir if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
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


def compare_mean_magnetization(simulations, output_dir=None):
    for simulation in simulations:
        if output_dir:
            simulation.show_lattice(f"{output_dir}/lattice_{simulation.name}_0.png")
        simulation.run(20)
        if output_dir:
            simulation.show_lattice(f"{output_dir}/lattice_{simulation.name}_1.png")
            simulation.save_results(f"{output_dir}/results_{simulation.name}_1.json")

    for simulation in simulations:
        plt.plot(
            range(len(simulation.info.mean_magnetization)),
            simulation.info.mean_magnetization,
            label=simulation.name.capitalize().replace("_", " "),
        )

    plt.xlabel("Sweeps")
    plt.ylabel("Mean magnetization")
    plt.legend()
    if output_dir:
        plt.savefig(f"{output_dir}/mean_magnetization.png", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def get_probability_of_complete_random(size, magnetization, attempts):
    p = (magnetization + 1) / 2
    p_star = sum(
        [math.comb(size, n) * 0.5**512 for n in range(int(p * size), size + 1)]
    )
    return sum(
        [
            p_star**i * (1 - p_star) ** (attempts - i) * math.comb(attempts, i)
            for i in range(1, attempts + 1)
        ]
    )


def run_thermalization_periods():
    run_thermalization_period(
        Method.METROPOLIS,
        FlipDynamics.SINGLE,
        "output/thermalization/metropolis/single",
    )
    run_thermalization_period(
        Method.METROPOLIS,
        FlipDynamics.COMPLETE_RANDOM_STEP,
        "output/thermalization/metropolis/complete",
    )
    run_thermalization_period(
        Method.GLAUBER,
        FlipDynamics.SINGLE,
        "output/thermalization/glauber/single",
    )
    run_thermalization_period(
        Method.GLAUBER,
        FlipDynamics.COMPLETE_RANDOM_STEP,
        "output/thermalization/glauber/complete",
    )


def plot_random_step_probability(size, attempts, output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    magnetization = np.linspace(0, 1, 100)
    plt.plot(
        magnetization,
        [get_probability_of_complete_random(size, m, attempts) for m in magnetization],
    )
    plt.xlabel("Magnetization")
    plt.ylabel("Finding Probability")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.axvline(x=0.13, color="red", linestyle="--", label="$M=0.13$")
    plt.legend()
    if output_dir:
        plt.savefig(f"{output_dir}/random_step_probability.png", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def compare_steps_to_magnetization(
    simulation_configurations,
    number_of_steps,
    magnetizations,
    number_of_simulations,
    output_file,
):
    random_lattices = [
        np.int8(
            np.random.choice(
                [-1, 1],
                size=(simulation_configurations[0]["length"],)
                * simulation_configurations[0]["dim"],
            )
        )
        for _ in range(number_of_simulations)
    ]
    for simulation_configuration in simulation_configurations:
        steps_to_reach = [0 for _ in range(len(magnetizations))]
        for random_lattice in random_lattices:
            simulation = Simulation(
                **simulation_configuration,
                initial_lattice=random_lattice,
            )
            simulation.run(number_of_steps)
            for i, magnetization in enumerate(magnetizations):
                steps_to_reach[
                    i
                ] += simulation.info.get_number_of_steps_to_mean_magnetization(
                    magnetization
                )
        steps_to_reach = [n / number_of_simulations for n in steps_to_reach]
        plt.plot(
            magnetizations,
            steps_to_reach,
            label=simulation.name.capitalize().replace("_", " "),
            marker="o",
            linestyle="",
        )

    plt.xlabel("Magnetization")
    plt.ylabel("Steps to Reach")
    plt.xticks([0, 0.3, 0.6, 0.99])
    plt.grid()
    plt.legend()
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_steps_to_magnetization(output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metropolis_simulation = {
        "name": "metropolis",
        "length": 512,
        "temperature": 1,
        "magnetic_field": 1,
        "magnetization_coefficient": 1,
        "dim": 1,
        "initial_state": InitialState.RANDOM,
        "method": Method.METROPOLIS,
        "flip_dynamics": FlipDynamics.SINGLE,
    }
    glauber_simulation = {
        "name": "glauber",
        "length": 512,
        "temperature": 1,
        "magnetic_field": 1,
        "magnetization_coefficient": 1,
        "dim": 1,
        "initial_state": InitialState.RANDOM,
        "method": Method.GLAUBER,
        "flip_dynamics": FlipDynamics.SINGLE,
    }
    compare_steps_to_magnetization(
        [metropolis_simulation, glauber_simulation],
        10000,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        100,
        output_file=f"{output_dir}/steps_to_magnetization.png" if output_dir else None,
    )


def run_chosen_simulation(output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation = Simulation(
        name="metropolis",
        length=512,
        temperature=1,
        magnetic_field=1,
        magnetization_coefficient=1,
        dim=1,
        initial_state=InitialState.UP,
        method=Method.METROPOLIS,
        flip_dynamics=FlipDynamics.SINGLE,
    )
    simulation.run(200, should_save=False)
    simulation.run(5000)
    plt.plot(simulation.info.mean_correlation)
    plt.xlabel("Distance")
    plt.ylabel("Correlation")
    if output_dir:
        plt.savefig(f"{output_dir}/correlation.png", bbox_inches="tight")
        plt.clf()
        with open(f"{output_dir}/final.json", "wt") as fd:
            fd.write(
                json.dumps(
                    {
                        "mean_energy": simulation.info.mean_energy,
                        "specific_heat": simulation.info.specific_heat,
                        "mean_energy_at_end_of_sweep": simulation.info.mean_energy_at_end_of_sweep,
                        "specific_heat_at_end_of_sweep": simulation.info.specific_heat_at_end_of_sweep,
                    },
                    indent=4,
                )
            )
    else:
        print(f"mean energy: {simulation.info.mean_energy}")
        print(f"specific heat: {simulation.info.specific_heat}")
        plt.show()


if __name__ == "__main__":
    # run_thermalization_periods()
    # plot_random_step_probability(512, 100, "output/random_step_probability")
    # plot_steps_to_magnetization("output/steps_to_magnetization")
    run_chosen_simulation("output/chosen_simulation")
