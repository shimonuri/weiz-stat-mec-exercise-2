import os
import numpy as np
import sympy
import json
from matplotlib import pyplot as plt


class BoltzmannSampler:
    def __init__(
        self,
        name,
        length,
        temperature,
        magnetic_field,
        magnetization_coefficient,
    ):
        self.name = name
        self.length = length
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.magnetization_coefficient = magnetization_coefficient

    def sample(self, number_of_samples):
        partition_function = 0
        magnetization_sums = {}
        magnetization_sum = 0
        for i in range(number_of_samples):
            if i % int(number_of_samples / 10) == 0:
                print(f"Sampled {i} times")

                if i != 0:
                    current_magnetization_sum = float(
                        magnetization_sum / partition_function
                    )
                    magnetization_sums[i] = current_magnetization_sum
                    print(f"Mean magnetization: {current_magnetization_sum}")

            lattice = np.random.choice([-1, 1], size=self.length)
            boltzmann_factor = sympy.exp(
                -self._get_lattice_energy(lattice) / self.temperature
            )
            partition_function += boltzmann_factor
            magnetization_sum += (
                self._get_lattice_magnetization(lattice) * boltzmann_factor
            )
        magnetization_sums[number_of_samples] = float(
            magnetization_sum / partition_function
        )
        return magnetization_sums

    def _get_lattice_magnetization(self, lattice):
        return np.sum(lattice) * self.magnetic_field / self.length

    def _get_lattice_energy(self, lattice):
        return -np.sum(lattice * np.roll(lattice, 1)) - self.magnetic_field * np.sum(
            lattice
        )


def main(length, temperature, number_of_samples, output_dir=None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler = BoltzmannSampler(
        name="Ising model",
        length=length,
        temperature=temperature,
        magnetic_field=1,
        magnetization_coefficient=1,
    )
    steps_to_magnetization = sampler.sample(number_of_samples)
    plt.plot(steps_to_magnetization.keys(), steps_to_magnetization.values())
    plt.xlabel("Number of samples")
    plt.ylabel("Magnetization")

    if output_dir is not None:
        with open(f"{output_dir}/result.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "length": length,
                        "temperature": temperature,
                        "magnetic_field": 1,
                        "magnetization_coefficient": 1,
                        "number_of_samples": number_of_samples,
                        "steps_to_magnetization": steps_to_magnetization,
                    },
                    indent=4,
                )
            )
        plt.savefig(f"{output_dir}/result.png")
        plt.clf()
    else:
        print(f"Mean steps_to_magnetization: {steps_to_magnetization}")
        plt.show()


if __name__ == "__main__":
    main(
        length=512,
        temperature=0.1,
        number_of_samples=10**4,
        output_dir="output/sampler/low_temperature",
    )
    main(
        length=512,
        temperature=10,
        number_of_samples=10**4,
        output_dir="output/sampler/high_temperature",
    )
