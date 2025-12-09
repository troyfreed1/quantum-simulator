"""
Quantum Circuit Simulator
Supports noiseless and noisy simulation of quantum circuits
"""

import re
import sys
import argparse
import numpy as np


class CircuitParser:
    """Parses quantum circuit files (.in format)"""

    def __init__(self, filename):
        self.filename = filename
        self.num_qubits = 0
        self.gates = []
        self.measurements = []

    def parse(self):
        """Parse circuit file and extract operations"""

        # Open the circuit file and read line by line
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                # Parse circuit size:
                if line.startswith('circuit:'):
                    match = re.search(r'circuit:\s*(\d+)\s*qubits?', line)
                    if match:
                        self.num_qubits = int(match.group(1))
                        print(f"Circuit has {self.num_qubits} qubits")

                # Parse measurement lines:
                elif line.startswith('measure'):
                    match = re.search(r'measure\s+(\d+)(?:\.\.(\d+))?', line)
                    if match:
                        start = int(match.group(1))
                        end = int(match.group(2)) if match.group(2) else start
                        self.measurements = list(range(start, end + 1))
                        print(f"Measure qubits: {self.measurements}")

                else:
                    # Try matching a single-qubit gate: ex, "H(0)"
                    single_match = re.match(r'([A-Z]+)\((\d+)\)', line)
                    if single_match:
                        self.gates.append({
                            'type': single_match.group(1),
                            'qubits': [int(single_match.group(2))]
                        })
                        continue

                    # Try matching a two-qubit gate: ex, "CNOT(0, 1)"
                    double_match = re.match(r'([A-Z]+)\((\d+),\s*(\d+)\)', line)
                    if double_match:
                        self.gates.append({
                            'type': double_match.group(1),
                            'qubits': [int(double_match.group(2)), int(double_match.group(3))]
                        })

        return self.num_qubits, self.gates, self.measurements


def optimize_circuit(gates):
    """Try optimizing cirucit by removing back-to-back canceling gates ex: (X, H, CNOT are self-inverse)"""
    optimized, i, removed = [], 0, 0

    while i < len(gates):
        if i + 1 < len(gates):
            g1, g2 = gates[i], gates[i + 1]
            if (g1['type'] == g2['type'] and g1['qubits'] == g2['qubits'] and
                g1['type'] in ['X', 'H', 'CNOT']):
                i += 2
                removed += 1
                continue
        optimized.append(gates[i])
        i += 1

    if removed > 0:
        print(f"Optimized: removed {removed} gate pairs ({len(gates)} -> {len(optimized)} gates)")
    return optimized


class QuantumSimulator:
    """Quantum circuit simulator with state vector representation"""

    def __init__(self, num_qubits, noise_prob=0.0, readout_fidelity=1.0, two_qubit_error=None):
        """Initialize quantum simulator

        Args:
            num_qubits: Number of qubits
            noise_prob: Single-qubit gate error probability
            readout_fidelity: Measurement accuracy (1.0 = perfect)
            two_qubit_error: 2-qubit gate error (default: 10x single-qubit) (true in real quantum hardware)
        """
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.single_qubit_error = noise_prob
        self.double_qubit_error = two_qubit_error if two_qubit_error is not None else noise_prob * 10
        self.readout_fidelity = readout_fidelity

        # Initializes the state vector |00...0⟩, use complex64 for memory efficiency rather than complex128
        self.state_vector = np.zeros(self.num_states, dtype=np.complex64)
        self.state_vector[0] = 1.0

        # Pre-compute gate matrices, faster than rebuilding gates each time
        sqrt_half = 1.0 / np.sqrt(2)
        self.X_gate = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        self.H_gate = sqrt_half * np.array([[1, 1], [1, -1]], dtype=np.complex64)

        # Fast random number generator better than np.random functions
        self.rng = np.random.Generator(np.random.PCG64())
        # Cache frequently-used values, stops repeated recomputation
        self.format_string = f'0{self.num_qubits}b'
        self.indices_cache = np.arange(self.num_states)

        print(f"Initialized {num_qubits}-qubit system (state vector: {self.num_states})")
        #print error rates
        if noise_prob > 0.0:
            print(f"Error rates: 1Q={self.single_qubit_error:.4f}, 2Q={self.double_qubit_error:.4f}")

    def get_state_vector(self):
        #Make a copy to prevent external modification while simulating
        return self.state_vector.copy()

    def apply_single_qubit_gate(self, gate_matrix, target_qubit):
        """Apply a 2x2 single-qubit gate to the full state vector.

        Groups amplitudes based on whether the target qubit
        is 0 or 1 in each basis state index. We then update both
        sets of amplitudes using the gate matrix. Using bit operations
        helps avoid reshaping the state vector which makes this much faster.
        """
        # Extract the four entries of the 2x2 gate matrix
        g00, g01 = gate_matrix[0, 0], gate_matrix[0, 1]
        g10, g11 = gate_matrix[1, 0], gate_matrix[1, 1]

        # For each index, check whether the target qubit bit is 0 or 1
        bit_values = (self.indices_cache >> target_qubit) & 1
        # Split indices into those where the target qubit is 0 or 1
        indices_0 = self.indices_cache[bit_values == 0]
        indices_1 = self.indices_cache[bit_values == 1]

        # Original amplitudes for the two groups
        psi_0, psi_1 = self.state_vector[indices_0], self.state_vector[indices_1]

        # Apply the gate matrix to update amplitudes
        self.state_vector[indices_0] = g00 * psi_0 + g01 * psi_1
        self.state_vector[indices_1] = g10 * psi_0 + g11 * psi_1

    #wrappers to avoid duplicate code
    def apply_X_gate(self, target_qubit):
        """Apply Pauli-X (NOT) gate"""
        self.apply_single_qubit_gate(self.X_gate, target_qubit)

    def apply_H_gate(self, target_qubit):
        """Apply Hadamard gate"""
        self.apply_single_qubit_gate(self.H_gate, target_qubit)

    def apply_CNOT_gate(self, control_qubit, target_qubit):
        """Apply the CNOT gate by changing the state vector.

        If the control qubit is 1, we flip the target qubit.
        If the control qubit is 0, we leave the state as is.
        We compute this for all basis states at once using bit tricks,
        making it more optimized rather than looping through each basis state.
        """

        # For each basis index, extract the bit of the control qubit (0 or 1)
        control_bit = (self.indices_cache >> control_qubit) & 1
        # Bit mask that flips ONLY the target qubit when XORed with the index
        toggle_mask = 1 << target_qubit
        # If control bit = 1  flip target bit
        # If control bit = 0  index stays the same
        new_indices = np.where(control_bit, self.indices_cache ^ toggle_mask, self.indices_cache)

        # Rearrange amplitudes according to the computed mapping
        self.state_vector = self.state_vector[new_indices]

    def apply_gate(self, gate):
        """Apply gate and noise (if enabled)"""
        # Extract gate type (e.g., "H", "X", "CNOT") and qubit indices
        gate_type, qubits = gate['type'], gate['qubits']

        if gate_type == 'X':
            self.apply_X_gate(qubits[0])
            self.apply_bit_flip_noise([qubits[0]], is_two_qubit=False)

        elif gate_type == 'H':
            self.apply_H_gate(qubits[0])
            self.apply_bit_flip_noise([qubits[0]], is_two_qubit=False)

        elif gate_type == 'CNOT':
            self.apply_CNOT_gate(qubits[0], qubits[1])
            self.apply_bit_flip_noise([qubits[0], qubits[1]], is_two_qubit=True)

        else:
            print(f" Unknown gate type {gate_type}")

    def measure(self, qubits_to_measure):
        """Measure qubits with readout errors
        Uses bit operations for efficiency. Also uses 1000 shots by default.
        """
        probabilities = self.state_vector.real**2 + self.state_vector.imag**2
        num_shots = 1000

        print(f"Measuring qubits {qubits_to_measure} ({num_shots} shots)")

        # Randomly sample basis state indices according to their probabilities
        measured_indices = self.rng.choice(self.num_states, size=num_shots, p=probabilities)
        results = {}

        # For each measurement, extract the qubits
        for idx in measured_indices:
            measured_value = 0
            for q in qubits_to_measure: # Read the q-th bit from the basis index
                bit = (idx >> q) & 1
                if self.rng.random() > self.readout_fidelity: # Apply readout noise
                    bit = 1 - bit
                measured_value = (measured_value << 1) | bit

            results[measured_value] = results.get(measured_value, 0) + 1

        # Convert to binary strings for output
        return {format(val, f'0{len(qubits_to_measure)}b'): count
                for val, count in results.items()}

    def measure_all(self):
        """Measure all qubits for convenience"""
        return self.measure(list(range(self.num_qubits)))

    def apply_bit_flip_noise(self, affected_qubits, is_two_qubit=False):
        """Apply bit-flip error with X gate based on error probabilities"""
        error = self.double_qubit_error if is_two_qubit else self.single_qubit_error
        if error == 0.0:
            return

        # Determine which qubits experience a random flip
        error_qubits = [q for q in affected_qubits if self.rng.random() < error]
        if error_qubits:
            print(f"Noise bit flips on qubits: {error_qubits}")
            for qubit in error_qubits:
                self.apply_X_gate(qubit)

    def print_state(self):
        """Print non-zero amplitudes of current state"""
        print("\n Current Quantum State:")
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                # Print |binary_state> : amplitude
                print(f"|{format(i, self.format_string)}> : {amplitude:.4f}")


def run_simulation(circuit_file, noise_mode=False, error_rate=0.0, custom_qubits=None, readout_fidelity=1.0, error_2q=None):
    """Load a circuit file, build a simulator, apply all gates, and perform measurements."""

    # Step 1: Parse the circuit file
    print(f"Loading circuit: {circuit_file}")
    parser = CircuitParser(circuit_file)
    num_qubits, gates, measurements = parser.parse()

    # Step 2: Allow user to override the number of qubits (optional)
    if custom_qubits is not None:
        if custom_qubits < num_qubits:
            print(f"Warning: Custom qubits ({custom_qubits}) < required ({num_qubits}), using {num_qubits}")
        else:
            num_qubits = custom_qubits

    print(f"\nCircuit: {num_qubits} qubits, {len(gates)} gates")
    print(f"Mode: {'Noise (error=' + str(error_rate) + ')' if noise_mode else 'Noiseless'}")

    # Step 3: Optimize circuit only when noise is off
    if not noise_mode:
        gates = optimize_circuit(gates)

    # Step 4: Initialize simulator with chosen noise settings
    print("\nSimulation")
    sim = QuantumSimulator(num_qubits,
                          noise_prob=error_rate if noise_mode else 0.0,
                          readout_fidelity=readout_fidelity,
                          two_qubit_error=error_2q)

    # Only prints intermediate states for small circuits
    verbose = num_qubits <= 5 and len(gates) <= 10

    print("\nApplying gates...")
    for i, gate in enumerate(gates, 1):
        if verbose:
            print(f"\n[Gate {i}/{len(gates)}]", end=" ")
        sim.apply_gate(gate)
        if verbose:
            sim.print_state()

    if not verbose:
        print(f"Applied {len(gates)} gates")

    # Step 5: Show final quantum state
    print("\n\nFinal State")
    sim.print_state()

    # Step 6: Perform measurements with cool visual
    if measurements:
        print("\nMeasurement")
        results = sim.measure(measurements)
        print("\nResults Summary")
        for state, count in sorted(results.items()):
            bar = '█' * int(count / 10)
            print(f"|{state}> : {count:4d} {bar}")
        return sim, results
    else:
        print("\nNo measurement specified")
        return sim, None


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python simulator.py -noiseless circuit.in\n"
               "  python simulator.py -noise circuit.in -error 0.01\n"
               "  python simulator.py -noiseless circuit.in -qubits 10"
    )

    parser.add_argument('circuit_file', help='Path to circuit input file (.in)')
    # User must choose exactly one mode: noiseless or noisy
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-noiseless', action='store_true', help='Run noiseless simulation')
    mode_group.add_argument('-noise', action='store_true', help='Run noisy simulation')
    # Optional settings
    parser.add_argument('-error', type=float, default=0.01,
                       help='Bit-flip error probability (default: 0.01)')
    parser.add_argument('-error2q', type=float, default=None,
                       help='Two-qubit gate error rate (default: 10x single-qubit)')
    parser.add_argument('-qubits', type=int, default=None,
                       help='Number of qubits (overrides circuit file)')
    parser.add_argument('-readout', type=float, default=1.0,
                       help='Readout fidelity (default: 1.0)')

    args = parser.parse_args()

    # Basic validation of user input
    if args.noise and not (0 <= args.error <= 1):
        parser.error("Error rate must be between 0 and 1")
    if args.qubits is not None and args.qubits < 1:
        parser.error("Number of qubits must be at least 1")
    if args.noiseless and args.error != 0.01:
        print("Warning: -error flag ignored in noiseless mode\n")

    # Try to run the simulation and catch common errors
    try:
        run_simulation(
            circuit_file=args.circuit_file,
            noise_mode=args.noise,
            error_rate=args.error if args.noise else 0.0,
            custom_qubits=args.qubits,
            readout_fidelity=args.readout,
            error_2q=args.error2q
        )
    except FileNotFoundError:
        print(f"Error: Circuit file '{args.circuit_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
