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
    def __init__(self, num_qubits, noise_prob=0.0):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.single_qubit_error = noise_prob
        self.double_qubit_error = noise_prob * 10
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0

        print(f"Initialized {num_qubits}-qubit system")
        print(f"State vector size: {self.num_states}")
        print(f"Initial state: |{'0' * num_qubits}>")
    # Return the state of the vector
    def get_state_vector(self):
        return self.state_vector.copy()
    # Apply a single qubit gate
    # We use this in our gate applications to apply the gates we designed
    def apply_single_qubit_gate(self, gate_matrix, target_qubit):
        n = self.num_qubits

        state = self.state_vector.reshape([2] * n)

        state = np.moveaxis(state, target_qubit, -1)

        state = np.tensordot(state, gate_matrix, axes=[[-1], [0]])

        state = np.moveaxis(state, -1, target_qubit)

        self.state_vector = state.flatten()
    # We apply the x gate by creating a complex array and using our apply
    # single qubit gate
    def apply_X_gate(self, target_qubit):
        X = np.array([[0,1],
                     [1,0]], dtype=complex)
        self.apply_single_qubit_gate(X, target_qubit)
        #print(f"Applied X gate to qubit {target_qubit}")
    # We do the same above for H gate
    def apply_H_gate(self, target_qubit):
        H = (1/np.sqrt(2)) * np.array([[1, 1],
                                       [1, -1]], dtype=complex)
        self.apply_single_qubit_gate(H, target_qubit)
        #print(f"Applied H gate to qubit {target_qubit}")
    # We do the same above for our CNOT
    def apply_CNOT_gate(self, control_qubit, target_qubit):
        new_state = np.zeros_like(self.state_vector)
        for i in range(self.num_states):
            binary = format(i, f'0{self.num_qubits}b')
            bits = [int(b) for b in binary]

            if bits[control_qubit] == 1:
                bits[target_qubit] = 1 - bits[target_qubit]
            new_index = int(''.join(map(str, bits)), 2)
            new_state[new_index] = self.state_vector[i]
        self.state_vector = new_state
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
        probabilities = np.abs(self.state_vector) ** 2

        num_shots = 1000
        results = {}

        print(f"Measuring qubits {qubits_to_measure} ({num_shots} shots)")

        # Measure the qubits for the amount of shots in the range of our total shots
        for shot in range(num_shots):
            measured_state_index = np.random.choice(
                self.num_states,
                p=probabilities
            )
            # We put our bits back into binary format
            full_binary = format(measured_state_index, f'0{self.num_qubits}b')

            measured_bits = ''.join([full_binary[q] for q in qubits_to_measure])
            # If the measured bit is in the results then we add by one else we set it to 1
            if measured_bits in results:
                results[measured_bits] += 1
            else: 
                results[measured_bits] = 1
            
        return results
    # Here we just measure all the qubits
    def measure_all(self):
        return self.measure(list(range(self.num_qubits)))
    # Here is where we apply quantum noise by using a random probability set in our noise_prob
    # By default it is set to 0.0
    def apply_bit_flip_noise(self, affected_qubits, is_two_qubit=False):
        error = self.double_qubit_error if is_two_qubit else self.single_qubit_error

        if error == 0.0:
            return

        for qubit in affected_qubits:
            if np.random.random() < error:
                print(f"Noise bit flip on qubit {qubit}")
                temp_noise = self.single_qubit_error
                self.single_qubit_error = 0.0
                self.apply_X_gate(qubit)
                self.single_qubit_error = temp_noise
         
    def print_state(self):
        print("\n Current Quantum State:")
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                binary_state = format(i, f'0{self.num_qubits}b')
                print(f"|{binary_state}> : {amplitude:.4f}")


# Test the parser with the example circuit
def run_simulation(circuit_file, noise_mode=False, error_rate=0.0):
    print(f"Loading circuit: {circuit_file}")
    parser = CircuitParser(circuit_file)

    num_qubits, gates, measurements = parser.parse()

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

    sim = QuantumSimulator(num_qubits, noise_prob=noise_prob)
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
        epilog="python simulator.py -noiseless circuit.in" \
        "python simulator.py -noise circuit.in -error 0.01" \
        "python simulator.py -noise circuit.in -error 0.1"
    )
    # We add our arguments here
    # The arguments are circuit_file, -noise or -noiseless and -error
    parser.add_argument('circuit_file', help='Path to circuit input file (.in)')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-noiseless', action='store_true', help='Run noiseless simulation')
    mode_group.add_argument('-noise', action='store_true', help='Run noisy simulation')

    parser.add_argument('-error', type=float, default=0.01, help='Bit-flip error probability (default: 0.01)')
    # Here we parse the cli arguments and then below use if blocks to trigger actions based on the flags
    args = parser.parse_args()

    if args.noise and not (0 <= args.error <= 1):
        parser.error("Error rate must be between 0 and 1")
    if args.noiseless and args.error != 0.01:
        print("Warning: -error flag ignored in noiseless mode\n")
    try:
        run_simulation(
            circuit_file=args.circuit_file,
            noise_mode=args.noise,
            error_rate=args.error if args.noise else 0.0
        )
    except FileNotFoundError:
        print(f"Error: Circuit file '{args.circuit_file} not found'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
