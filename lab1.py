import numpy as np
import math

def trapezoidal_rule(f, a, b, n, *args):
    """Approximate the integral of f from a to b using the trapezoidal rule."""
    x = np.linspace(a, b, n+1)
    y = f(x, *args)
    h = (b - a) / n
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def f_even(x, n):
    """Function for even student numbers: f(x) = x^n * exp(-x^2 / n)"""
    return x*n * np.exp(-x**2 / n)

def f_odd(x, n):
    """Function for odd student numbers: f(x) = n * sin(πnx)"""
    return n * np.sin(np.pi * n * x)

def a_k(f, k, n, interval, num_points=1000):
    """Compute the Fourier coefficient a_k."""
    a, b = interval
    return (2 / (b - a)) * trapezoidal_rule(lambda x: f(x, n) * np.cos(k * x), a, b, num_points)

def b_k(f, k, n, interval, num_points=1000):
    """Compute the Fourier coefficient b_k."""
    a, b = interval
    return (2 / (b - a)) * trapezoidal_rule(lambda x: f(x, n) * np.sin(k * x), a, b, num_points)

def fourier_series_approximation(f, n, N, x, interval):
    """Compute the Fourier series approximation."""
    a_0 = a_k(f, 0, n, interval) / 2
    series_sum = a_0
    for k in range(1, N + 1):
        series_sum += a_k(f, k, n, interval) * np.cos(k * x) + b_k(f, k, n, interval) * np.sin(k * x)

    if n % 2 == 1:
        return series_sum / 2
    else:
        return series_sum


def plot_fourier_series(f, n, N, interval):
    """Plot the Fourier series approximation."""
    import matplotlib.pyplot as plt
    x_values = np.linspace(interval[0], interval[1], 1000)
    y_values = [fourier_series_approximation(f, n, N, xi, interval) for xi in x_values]

    plt.plot(x_values, y_values, label=f'Fourier Series Approximation (N={N})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Fourier Series Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()
def relative_error(f, f_approx, n, interval, num_points=1000):
    """Calculate the relative error between the original function and its Fourier series approximation."""
    x_values = np.linspace(interval[0], interval[1], num_points)
    errors = np.abs(f(x_values, n) - [f_approx(xi) for xi in x_values]) / np.abs(f(x_values, n))
    return np.mean(errors)
def main():
    n = 13  # Replace with your student number
    N = 200  # Order of approximation
    interval = (-np.pi, np.pi) if n % 2 == 0 else (0, np.pi)
    f = f_even if n % 2 == 0 else f_odd

    plot_fourier_series(f, n, N, interval)
    # Additional code for saving results, calculating error, etc.

if __name__ == '__main__':
    main()