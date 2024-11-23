import math
import random
import datetime
import time
import matplotlib.pyplot as plt

def generate_signal(N):
    random.seed(datetime.datetime.now().timestamp())
    signal = [random.random() for _ in range(N)]
    return signal

def dft(signal):
    N = len(signal)
    A, B = [], []
    operations = 0
    add_ops = 0
    mult_ops = 0
    for k in range(N):
        a = 0
        b = 0
        for i in range(N):
            a += signal[i] * math.cos(2 * math.pi * k * i / N)
            b += signal[i] * math.sin(2 * math.pi * k * i / N)
            add_ops += 2  
            mult_ops += 2 
        a = a / N
        b = b / N
        A.append(a)
        B.append(b)
        operations += 2*(4*N + N + 1)
        add_ops += 2
    return A, B, operations, add_ops, mult_ops

def fft(signal):
    N = len(signal)
    operations = 0
    add_ops = 0
    mult_ops = 0

    if N <= 1:
        return signal, [0], operations, add_ops, mult_ops

    even_signal = signal[0::2]
    odd_signal = signal[1::2]

    even_A, even_B, even_operations, even_add_ops, even_mult_ops = fft(even_signal)
    odd_A, odd_B, odd_operations, odd_add_ops, odd_mult_ops = fft(odd_signal)

    A = [0 for _ in range(N)]
    B = [0 for _ in range(N)]

    for k in range(N // 2):
        twiddle_real = math.cos(-2 * math.pi * k / N)
        twiddle_imag = math.sin(-2 * math.pi * k / N)

        real_part = odd_A[k] * twiddle_real - odd_B[k] * twiddle_imag
        imag_part = odd_A[k] * twiddle_imag + odd_B[k] * twiddle_real

        A[k] = even_A[k] + real_part
        B[k] = even_B[k] + imag_part

        A[k + N // 2] = even_A[k] - real_part
        B[k + N // 2] = even_B[k] - imag_part

        operations += 14
        add_ops += 6  
        mult_ops += 4 

    operations += even_operations + odd_operations + 2 * N
    add_ops += even_add_ops + odd_add_ops
    mult_ops += even_mult_ops + odd_mult_ops

    return A, B, operations, add_ops, mult_ops

def run_and_measure(N_values):
    dft_times, fft_times = [], []
    dft_ops, fft_ops = [], []
    dft_add_ops, fft_add_ops = [], []
    dft_mult_ops, fft_mult_ops = [], []

    for N in N_values:
        signal = generate_signal(N)

        start_time = time.time()
        dft_A, dft_B, dft_operations, dft_add, dft_mult = dft(signal)
        end_time = time.time()
        dft_times.append(end_time - start_time)
        dft_ops.append(dft_operations)
        dft_add_ops.append(dft_add)
        dft_mult_ops.append(dft_mult)

        start_time = time.time()
        fft_A, fft_B, fft_operations, fft_add, fft_mult = fft(signal)
        end_time = time.time()
        fft_times.append(end_time - start_time)
        fft_ops.append(fft_operations)
        fft_add_ops.append(fft_add)
        fft_mult_ops.append(fft_mult)

        print(f"\nN = {N}:")
        print(f"DFT: Time = {dft_times[-1]:.6f} sec, Operations = {dft_operations}, Add Ops = {dft_add}, Mult Ops = {dft_mult}")
        print(f"FFT: Time = {fft_times[-1]:.6f} sec, Operations = {fft_operations}, Add Ops = {fft_add}, Mult Ops = {fft_mult}")

    return dft_times, fft_times, dft_ops, fft_ops, dft_add_ops, fft_add_ops, dft_mult_ops, fft_mult_ops

def plot_results(N_values, dft_ops, fft_ops, dft_times, fft_times, dft_add_ops, fft_add_ops, dft_mult_ops, fft_mult_ops):
    plt.figure(figsize=(14, 8))

    # Час виконання DFT та FFT
    plt.subplot(2, 2, 1)
    plt.plot(N_values, dft_times, label='DFT Time', marker='o')
    plt.plot(N_values, fft_times, label='FFT Time', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('Computation Time for DFT and FFT')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(N_values, dft_ops, label='DFT Operations', marker='o')
    plt.plot(N_values, fft_ops, label='FFT Operations', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Number of Operations')
    plt.yscale('log')
    plt.title('DFT vs FFT Operations')
    plt.legend()

    # Операції додавання
    plt.subplot(2, 2, 3)
    plt.plot(N_values, dft_add_ops, label='DFT Add Ops', marker='o')
    plt.plot(N_values, fft_add_ops, label='FFT Add Ops', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Addition Operations Count')
    plt.yscale('log')
    plt.title('Addition Operations Count for DFT and FFT')
    plt.legend()

    # Операції множення
    plt.subplot(2, 2, 4)
    plt.plot(N_values, dft_mult_ops, label='DFT Mul Ops', marker='o')
    plt.plot(N_values, fft_mult_ops, label='FFT Mul Ops', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Multiplication Operations Count')
    plt.yscale('log')
    plt.title('Multiplication Operations Count for DFT and FFT')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    N_values = [2**i for i in range(1, 10)]
    dft_times, fft_times, dft_ops, fft_ops, dft_add_ops, fft_add_ops, dft_mult_ops, fft_mult_ops = run_and_measure(N_values)
    plot_results(N_values, dft_ops, fft_ops, dft_times, fft_times, dft_add_ops, fft_add_ops, dft_mult_ops, fft_mult_ops)

if __name__ == "__main__":
    main()