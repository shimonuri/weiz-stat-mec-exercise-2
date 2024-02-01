import os
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 25})


def plot_zeros(magnetization_coefficient, number_of_spins, output_dir=None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zeros = get_zeros(magnetization_coefficient, number_of_spins)

    plt.plot(np.real(zeros), np.imag(zeros), "o")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    if output_dir is not None:
        plt.savefig(
            os.path.join(output_dir, f"zeros_{number_of_spins}.png"),
            bbox_inches="tight",
        )
        plt.clf()
    else:
        plt.show()


def plot_convergence_to_zero(magnetization_coefficient, output_dir=None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    minimal_distance = []
    numbers_of_spins = list(range(30, 100000, 10000))
    for number_of_spins in numbers_of_spins:
        zeros = get_zeros(magnetization_coefficient, number_of_spins)
        minimal_distance.append(min([abs(zero) for zero in zeros]))

    plt.plot(numbers_of_spins, minimal_distance, "o")
    plt.xlabel("$N$")
    plt.ylabel("$|z_{min}|$")
    plt.semilogy()
    if output_dir is not None:
        plt.savefig(
            os.path.join(output_dir, f"convergence_to_zero.png"),
            bbox_inches="tight",
        )
        plt.clf()
    else:
        plt.show()


def get_zeros(magnetization_coefficient, number_of_spins):
    zeros = []
    for n in range(4 * magnetization_coefficient):
        n_exp = np.exp(2j * np.pi / (4 * magnetization_coefficient) * n)
        for m in range(number_of_spins):
            m_exp = np.exp(2j * np.pi * m / number_of_spins)
            zero = n_exp * (
                (m_exp * np.exp(1j * np.pi / number_of_spins) + 1)
                / (m_exp * np.exp(1j * np.pi / number_of_spins) - 1)
            ) ** (1 / (2 * magnetization_coefficient))
            zeros.append(zero)
    return zeros


if __name__ == "__main__":
    # plot_zeros(
    #     magnetization_coefficient=1, number_of_spins=1000, output_dir="output/yang_lee/"
    # )
    plot_convergence_to_zero(magnetization_coefficient=1, output_dir="output/yang_lee/")
