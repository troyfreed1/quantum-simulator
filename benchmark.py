import time
import sys
import os
import numpy as np
from simulator import CircuitParser, QuantumSimulator
# Here we measure the amount of time it takes to run the amount of qubits
def benchmark_qubits():
    print("Benchmark 1: Runtime vs Number of Qubits")

    print("Circuit: apply H gate to all qubits")

    qubits_counts = [3,5,7,10,12,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    results = []

    for n in qubits_counts:
        start_time = time.time()
        sim = QuantumSimulator(n, noise_prob=0.0)

        for qubit in range(n):
            sim.apply_H_gate(qubit)

        sim.measure_all()

        end_time = time.time()
        elapsed = end_time - start_time

        memory_mb = (2 ** n) * 8 / (1024 ** 2)  # complex64 = 8 bytes

        print(f"Qubits: {n:<10} Possible Qubit Comb: {2**n:<15} Elapsed: {elapsed:<12.4f} Memory Used:{memory_mb:<12.2f}")
        results.append((n, 2**n, elapsed, memory_mb))
    print()
    return results
# This is where we measure the amount of time it takes to run 10 qubits on a different amount of gates
def benchmark_gates():
    print("Benchmark 2: Runtime vs Number of gates")

    print("Circuit: 10 qubits, varying number of H gates")

    n_qubits = 10
    gate_counts = [10,50,100,200,500,1000]

    results = []

    print(f"{'Gates':<10} {'Time (s)':<12}")
    print("\n")

    for num_gates in gate_counts:
        start_time = time.time()
        sim = QuantumSimulator(n_qubits, noise_prob=0.0)

        for i in range(num_gates):
            qubit = i % n_qubits
            sim.apply_H_gate(qubit)
        end_time = time.time()
        elapsed = end_time - start_time

        print(f"{num_gates:<10} {elapsed:<.4f}")
        results.append((num_gates, elapsed))
    print()
    return results
# Here we test the time it takes to run our benchmark files
def benchmark_circuit_file(circuit_file):
    print(f"\nBenchmarking circuit: {circuit_file}")

    parser = CircuitParser(circuit_file)
    num_qubits, gates, measurements = parser.parse()

    start_time = time.time()

    sim = QuantumSimulator(num_qubits, noise_prob=0.0)

    for gate in gates:
        sim.apply_gate(gate)
    if measurements:
        sim.measure(measurements)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Qubits {num_qubits}, Gates {len(gates)}, States {2**num_qubits}")

    return elapsed
if __name__ == "__main__":
    # this is where we actually run the benchmarks
    print("Quantum Simulator Performance Benchmarks")

    print("\n\nStarting benchmarks:")

    qubit_results = benchmark_qubits()
    gate_results = benchmark_gates()

    test_circuits = [
        'test_bell.in',
        'test_circuit.in',
        'test_ghz.in'
    ]

    print("Benchmark 3: Test circuit performance")

    for circuit in test_circuits:
        if os.path.exists(circuit):
            benchmark_circuit_file(circuit)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nComplexity Analysis:")
    print("- State vector size: O(2^n) where n = number of qubits")
    print("- Single-qubit gate: O(2^n) matrix-vector multiplication")
    print("- Two-qubit gate: O(2^n) state transformation")
    print("- Memory usage: O(2^n) complex numbers (complex64 = 8 bytes each)")
    print("\nScalability:")
    if len(qubit_results) >= 2:
        # Compare last two measurements
        n1, states1, time1, mem1 = qubit_results[-2]
        n2, states2, time2, mem2 = qubit_results[-1]
        ratio = time2 / time1
        print(f"- Going from {n1} to {n2} qubits: ~{ratio:.1f}x slower")
        print(f"- Memory grows exponentially: {mem2/mem1:.1f}x increase")
