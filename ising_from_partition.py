import sympy
import numpy as np


def main_sympy(number_of_spins, magnetic_field, magnetization_coefficient, temperature):
    beta = sympy.symbols("beta")

    lamda_plus = sympy.cosh(magnetic_field * beta) * sympy.exp(
        beta * magnetization_coefficient
    ) + sympy.sqrt(
        sympy.exp(2 * beta * magnetization_coefficient)
        * sympy.sinh(magnetic_field * beta) ** 2
        + sympy.exp(-2 * beta * magnetization_coefficient)
    )
    lamda_minus = sympy.cosh(magnetic_field * beta) * sympy.exp(
        beta * magnetization_coefficient
    ) - sympy.sqrt(
        sympy.exp(2 * beta * magnetization_coefficient)
        * sympy.sinh(magnetic_field * beta) ** 2
        + sympy.exp(-2 * beta * magnetization_coefficient)
    )
    partition_function = lamda_plus**number_of_spins + lamda_minus**number_of_spins
    log_partition_function = sympy.log(partition_function)
    energy = -sympy.diff(log_partition_function, beta)
    heat_capacity = sympy.diff(energy, beta) * beta**2
    correlation_fu
    print(f"Energy: {float(energy.subs({beta: 1/temperature}))}")
    print(f"Specific heat capacity: {float(heat_capacity.subs({beta: 1/temperature}))}")
    print(
        f"Correlation length: {float(correlation_length.subs({beta: 1/temperature}))}"
    )


if __name__ == "__main__":
    main_sympy(
        number_of_spins=512,
        magnetic_field=1,
        magnetization_coefficient=1,
        temperature=1,
    )
