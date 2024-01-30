import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 25})


def plot_zeros(magnetization_coefficient, number_of_spins):
    zeros = []
    for n in range(4 * magnetization_coefficient):
        n_exp = np.exp(2j * np.pi / (4 * magnetization_coefficient) * n)
        for m in range(number_of_spins):
            m_exp = np.exp(2j * np.pi * m / number_of_spins)
            print((n, m))
            zero = n_exp * (
                (m_exp * np.exp(1j * np.pi / number_of_spins) + 1)
                / (m_exp * np.exp(1j * np.pi / number_of_spins) - 1)
            ) ** (1 / (2 * magnetization_coefficient))
            zeros.append(zero)

    # plot zeros
    plt.plot(np.real(zeros), np.imag(zeros), "o")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.show()


if __name__ == "__main__":
    plot_zeros(magnetization_coefficient=1, number_of_spins=100)
