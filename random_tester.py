import random
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 25})


def plot_correlation(number_of_samples, max_k, output_file=None):
    samples = [random.random() for _ in range(number_of_samples)]
    k = range(1, max_k + 1)
    correlations = [get_correlation(samples, i) / (1 / 4) for i in k]
    plt.plot(k, correlations)
    plt.xlabel("k")
    plt.ylabel(r"$\langle x_ix_{i+k} \rangle / (1/4)$")
    # xtick format
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_correlation_var(number_of_samples, max_k, output_file=None):
    samples = [random.random() for _ in range(number_of_samples)]
    k = range(1, max_k + 1)
    correlations = [get_correlation_var(samples, i) / (7 / 144) for i in k]
    plt.plot(k, correlations)
    # xtick format
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel("k")
    plt.ylabel(r"$Var({x_ix_{i+k}}) / (7/144)$")
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def get_correlation_var(samples, k):
    return (
        sum((samples[i] * samples[i + k]) ** 2 for i in range(len(samples) - k))
        / (len(samples) - k)
        - get_correlation(samples, k) ** 2
    )


def get_correlation(samples, k):
    return sum(samples[i] * samples[i + k] for i in range(len(samples) - k)) / (
        len(samples) - k
    )


if __name__ == "__main__":
    plot_correlation(10**6, 50, "output/correlation.png")
    plot_correlation_var(10**6, 50, "output/correlation_var.png")
