"""
Quantum Circuit Simulator
Supports noiseless and noisy simulation of quantum circuits
"""

import re
import sys
import argparse
import numpy as np


class CircuitParser:
    """
    Parses quantum circuit files in the custom .in format

    Expected format:
        circuit: N qubits
        GATE(qubit_index)
        GATE(control, target)
        measure start..end
    """

    def __init__(self, filename):
        self.filename = filename
        self.num_qubits = 0
        self.gates = []
        self.measurements = []

    def parse(self):
        """Parse the circuit file and extract all operations"""
        with open(self.filename, 'r') as f:
            for line in f:
                # Remove whitespace and skip comments/empty lines
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                # Parse circuit declaration: "circuit: N qubits"
                if line.startswith('circuit:'):
                    match = re.search(r'circuit:\s*(\d+)\s*qubits?', line)
                    if match:
                        self.num_qubits = int(match.group(1))
                        print(f"Circuit has {self.num_qubits} qubits")

                # Parse measurement: "measure 0..1" or "measure 0"
                elif line.startswith('measure'):
                    match = re.search(r'measure\s+(\d+)(?:\.\.(\d+))?', line)
                    if match:
                        start = int(match.group(1))
                        end = int(match.group(2)) if match.group(2) else start
                        self.measurements = list(range(start, end + 1))
                        print(f"Measure qubits: {self.measurements}")

                # Parse gates: X(0), H(1), CNOT(0,1)
                else:
                    # Match single-qubit gates: GATE(qubit)
                    single_match = re.match(r'([A-Z]+)\((\d+)\)', line)
                    if single_match:
                        gate_name = single_match.group(1)
                        qubit = int(single_match.group(2))
                        self.gates.append({
                            'type': gate_name,
                            'qubits': [qubit]
                        })
                        print(f"Added gate: {gate_name} on qubit {qubit}")
                        continue

                    # Match two-qubit gates: GATE(qubit1, qubit2)
                    double_match = re.match(r'([A-Z]+)\((\d+),\s*(\d+)\)', line)
                    if double_match:
                        gate_name = double_match.group(1)
                        qubit1 = int(double_match.group(2))
                        qubit2 = int(double_match.group(3))
                        self.gates.append({
                            'type': gate_name,
                            'qubits': [qubit1, qubit2]
                        })
                        print(f"Added gate: {gate_name} on qubits {qubit1}, {qubit2}")

        return self.num_qubits, self.gates, self.measurements

class QuantumSimulator:
    def __init__(self, num_qubits, noise_prob=0.0):
        self.num_qubits = num_qubits
        self.noise_prob = noise_prob
        self.num_states = 2 ** num_qubits
        
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0

        print(f"Initialized {num_qubits}-qubit system")
        print(f"State vector size: {self.num_states}")
        print(f"Initial state: |{'0' * num_qubits}>")
    def get_state_vector(self):
        return self.state_vector.copy()
    def print_state(self):
        print("\n Current Quantum State:")
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                binary_state = format(i, f'0{self.num_qubits}b')
                print(f"|{binary_state}> : {amplitude:.4f}")


# Test the parser with the example circuit
if __name__ == "__main__":
    # For now, let's test the parser
    print("=== Testing Circuit Parser ===\n")

    # We'll create a test circuit file first
    test_circuit = """// Deutsch-Jozsa for one of
// the balanced f : {0,1}^2 -> {0,1}^2
circuit: 3 qubits
X(2)
H(0)
H(1)
H(2)
// U_f
CNOT(0,2)
H(0)
H(1)
measure 0..1
"""

    # Write test circuit
    with open('test_circuit.in', 'w') as f:
        f.write(test_circuit)

    # Parse it
    parser = CircuitParser('test_circuit.in')
    num_qubits, gates, measurements = parser.parse()

    print(f"\n=== Parsing Complete ===")
    print(f"Total qubits: {num_qubits}")
    print(f"Total gates: {len(gates)}")
    print(f"Gates to apply: {gates}")
    print(f"Qubits to measure: {measurements}")

    print('\nTesting Quantum Simulator:')
    sim = QuantumSimulator(num_qubits)
    sim.print_state()
