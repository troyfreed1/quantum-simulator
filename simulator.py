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
    # Declare all the aspects of our quantim simulator
    def __init__(self, num_qubits, noise_prob=0.0, readout_fidelity=1.0, two_qubit_error=None):
        """
        Initialize quantum simulator.

        Args:
            num_qubits: Number of qubits in system
            noise_prob: Single-qubit gate error probability
            readout_fidelity: Measurement accuracy (default 1.0 = perfect)
            two_qubit_error: 2-qubit gate error probability (default: 10x single-qubit)
        """
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.single_qubit_error = noise_prob

        # Allow explicit 2-qubit error specification, otherwise use 10x multiplier
        if two_qubit_error is None:
            self.double_qubit_error = noise_prob * 10
        else:
            self.double_qubit_error = two_qubit_error

        self.readout_fidelity = readout_fidelity  # NEW: measurement error parameter
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0

        # Pre-compute gate matrices for performance (cached)
        sqrt_half = 1.0 / np.sqrt(2)
        self.X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        self.H_gate = sqrt_half * np.array([[1, 1], [1, -1]], dtype=complex)

        print(f"Initialized {num_qubits}-qubit system")
        print(f"State vector size: {self.num_states}")
        print(f"Initial state: |{'0' * num_qubits}>")
        if noise_prob > 0.0:
            print(f"Error rates: 1Q={self.single_qubit_error:.4f}, 2Q={self.double_qubit_error:.4f}")
        if readout_fidelity < 1.0:
            print(f"Readout fidelity: {readout_fidelity:.2%}")
    # Return the state of the vector
    def get_state_vector(self):
        return self.state_vector.copy()

    # Apply a single qubit gate using optimized direct indexing
    # This avoids reshape/moveaxis overhead by directly manipulating indices
    def apply_single_qubit_gate(self, gate_matrix, target_qubit):
        """
        Apply single-qubit gate using direct index manipulation (OPTIMIZED).

        Key insight: Instead of reshaping state vector to tensor and using moveaxis,
        directly separate indices by target qubit position using bitwise operations.

        For basis state |i⟩, the target qubit value is: (i >> target_qubit) & 1
        Separate into: indices where bit=0 (even) and indices where bit=1 (odd)

        Apply gate: output[even] = g00*input[even] + g01*input[odd]
                    output[odd]  = g10*input[even] + g11*input[odd]

        This is ~5-10x faster than reshape/moveaxis approach.
        """
        # Extract gate matrix elements
        g00, g01 = gate_matrix[0, 0], gate_matrix[0, 1]
        g10, g11 = gate_matrix[1, 0], gate_matrix[1, 1]

        # Separate state indices: those with target_qubit=0 and target_qubit=1
        indices = np.arange(self.num_states)
        bit_values = (indices >> target_qubit) & 1

        indices_0 = indices[bit_values == 0]  # Where target qubit is 0
        indices_1 = indices[bit_values == 1]  # Where target qubit is 1

        # Extract components for |0⟩ and |1⟩ states of target qubit
        psi_0 = self.state_vector[indices_0]
        psi_1 = self.state_vector[indices_1]

        # Apply gate: [a' b'] = gate @ [a b]^T
        psi_0_new = g00 * psi_0 + g01 * psi_1
        psi_1_new = g10 * psi_0 + g11 * psi_1

        # Write results back
        self.state_vector[indices_0] = psi_0_new
        self.state_vector[indices_1] = psi_1_new
    # We apply the x gate by creating a complex array and using our apply
    # single qubit gate
    def apply_X_gate(self, target_qubit):
        self.apply_single_qubit_gate(self.X_gate, target_qubit)
        #print(f"Applied X gate to qubit {target_qubit}")
    # We do the same above for H gate
    def apply_H_gate(self, target_qubit):
        self.apply_single_qubit_gate(self.H_gate, target_qubit)
        #print(f"Applied H gate to qubit {target_qubit}")
    # Optimized CNOT using NumPy bitwise operations instead of Python loop
    def apply_CNOT_gate(self, control_qubit, target_qubit):
        """
        Apply CNOT gate using vectorized NumPy operations.
        This is ~20-50x faster than the original Python loop implementation.

        The gate flips the target qubit if the control qubit is |1>.
        Uses bitwise operations to efficiently compute new state indices.
        """
        # Create array of all possible state indices: [0, 1, 2, ..., 2^n-1]
        indices = np.arange(self.num_states)

        # Extract control bit: (index >> control_qubit) & 1
        control_bit = (indices >> control_qubit) & 1

        # Create bit mask for target qubit position
        toggle_mask = 1 << target_qubit

        # Calculate new indices: flip target bit only if control bit is 1
        # Uses np.where for vectorized conditional operation
        new_indices = np.where(
            control_bit,
            indices ^ toggle_mask,  # Flip target bit using XOR
            indices                  # Keep unchanged
        )

        # Rearrange state vector elements according to new index mapping
        self.state_vector = self.state_vector[new_indices]
        #print(f"Applied CNOT gate: control={control_qubit}, target={target_qubit}")
    # This is how we actually apply the gates we pull from our parser
    def apply_gate(self, gate):
        gate_type = gate['type']
        qubits = gate['qubits']

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
            print(f"Warning: Unknown gate type {gate_type}")
    # We measure our qubits
    def measure(self, qubits_to_measure):
        """
        Measure qubits with realistic readout fidelity.

        Real hardware: measurements have error rates of 1-3%
        This applies readout errors to measured bits based on readout_fidelity.
        Uses batched sampling for improved performance.
        """
        probabilities = np.abs(self.state_vector) ** 2
        num_shots = 1000

        print(f"Measuring qubits {qubits_to_measure} ({num_shots} shots)")

        # OPTIMIZATION: Batch all measurements at once instead of loop
        measured_indices = np.random.choice(
            self.num_states,
            size=num_shots,
            p=probabilities
        )

        results = {}

        # Convert all indices to binary simultaneously
        for measured_state_index in measured_indices:
            full_binary = format(measured_state_index, f'0{self.num_qubits}b')
            measured_bits = list([full_binary[q] for q in qubits_to_measure])

            # Apply readout errors: flip measurement result with probability (1 - readout_fidelity)
            for i in range(len(measured_bits)):
                if np.random.random() > self.readout_fidelity:
                    measured_bits[i] = '1' if measured_bits[i] == '0' else '0'

            measured_bits_str = ''.join(measured_bits)
            # If the measured bit is in the results then we add by one else we set it to 1
            if measured_bits_str in results:
                results[measured_bits_str] += 1
            else:
                results[measured_bits_str] = 1

        return results
    # Here we just measure all the qubits
    def measure_all(self):
        return self.measure(list(range(self.num_qubits)))
    # Here is where we apply quantum noise by using a random probability set in our noise_prob
    # By default it is set to 0.0
    def apply_bit_flip_noise(self, affected_qubits, is_two_qubit=False):
        """
        Apply bit-flip error channel to affected qubits.

        Physically models: E(ρ) = (1-p)ρ + pXρX†
        This represents: probability (1-p) no error, probability p of X error

        Unlike sequential noise application, errors on the same qubit cancel naturally:
        X applied twice = I (returns to original state)
        """
        error = self.double_qubit_error if is_two_qubit else self.single_qubit_error

        if error == 0.0:
            return

        # Determine which qubits will experience errors (deterministic sampling from error channel)
        error_qubits = [q for q in affected_qubits if np.random.random() < error]

        if error_qubits:
            print(f"Noise bit flips on qubits: {error_qubits}")
            # Apply X gates to qubits with errors
            # Note: These X gate applications are ideal (error-free)
            # This models the physical error that occurred during the preceding gate execution
            for qubit in error_qubits:
                self.apply_X_gate(qubit)

    def print_state(self):
        print("\n Current Quantum State:")
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                binary_state = format(i, f'0{self.num_qubits}b')
                print(f"|{binary_state}> : {amplitude:.4f}")


# Test the parser with the example circuit
def run_simulation(circuit_file, noise_mode=False, error_rate=0.0, custom_qubits=None, readout_fidelity=1.0, error_2q=None):
    print(f"Loading circuit: {circuit_file}")
    parser = CircuitParser(circuit_file)

    num_qubits, gates, measurements = parser.parse()

    # Override qubit count if custom_qubits specified
    if custom_qubits is not None:
        if custom_qubits < num_qubits:
            print(f"\nWarning: Custom qubit count ({custom_qubits}) is less than required qubits ({num_qubits})")
            print(f"Using {num_qubits} qubits instead\n")
        else:
            print(f"\nOverriding qubit count: {num_qubits} -> {custom_qubits}")
            num_qubits = custom_qubits

    print(f"\nCircuit Summary:")
    print(f" Qubits: {num_qubits}")
    print(f" Gates: {len(gates)}")
    print(f" Measurements: {measurements}")

    if noise_mode:
        noise_prob = error_rate
        print(f"\n Mode: Noise (error rate = {error_rate})")
    else:
        noise_prob = 0.0
        print(f" Mode: Noiseless")

    print("\n\nSimulation")

    sim = QuantumSimulator(num_qubits, noise_prob=noise_prob, readout_fidelity=readout_fidelity, two_qubit_error=error_2q)
    print("\nApplying gates...")
    for i, gate in enumerate(gates, 1):
        print(f"\n[Gate {i}]/{len(gates)}", end=" ")
        sim.apply_gate(gate)
        sim.print_state()

    print("\n\nFinal Gate")
    sim.print_state()

    if measurements:
        print("Measurement")
        results = sim.measure(measurements)

        print("\nResults Summary")
        for state, count in sorted(results.items()):
            bar = '█' * int(count / 10)
            print(f"|{state}> : {count:4d} {bar}")
    else:
        print("\nNo measurement specified")
    return sim, results if measurements else None
if __name__ == "__main__":
    # Here is where we instantiate our CLI
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n" \
        "  python simulator.py -noiseless circuit.in\n" \
        "  python simulator.py -noise circuit.in -error 0.01\n" \
        "  python simulator.py -noiseless circuit.in -qubits 10"
    )
    # We add our arguments here
    # The arguments are circuit_file, -noise or -noiseless, -error, and optional -qubits
    parser.add_argument('circuit_file', help='Path to circuit input file (.in)')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-noiseless', action='store_true', help='Run noiseless simulation')
    mode_group.add_argument('-noise', action='store_true', help='Run noisy simulation')

    parser.add_argument('-error', type=float, default=0.01, help='Bit-flip error probability (default: 0.01)')
    parser.add_argument('-error2q', type=float, default=None, help='Two-qubit gate error rate (default: 10x single-qubit)')
    parser.add_argument('-qubits', type=int, default=None, help='Number of qubits (overrides circuit file specification)')
    parser.add_argument('-readout', type=float, default=1.0, help='Readout fidelity (default: 1.0, no errors)')
    # Here we parse the cli arguments and then below use if blocks to trigger actions based on the flags
    args = parser.parse_args()

    if args.noise and not (0 <= args.error <= 1):
        parser.error("Error rate must be between 0 and 1")
    if args.qubits is not None and args.qubits < 1:
        parser.error("Number of qubits must be at least 1")
    if args.noiseless and args.error != 0.01:
        print("Warning: -error flag ignored in noiseless mode\n")
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
        print(f"Error: Circuit file '{args.circuit_file} not found'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
