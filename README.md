# Quantum Circuit Simulator

A high-performance quantum circuit simulator supporting both noiseless and noisy simulation modes. Handles circuits with up to 30 qubits on typical hardware.

## Requirements

- Python 3.7+
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/troyfreed1/quantum-simulator
cd QuantumSim

# Install NumPy
pip install numpy
```

## Quick Start

**Run a Bell state circuit:**
```bash
python simulator.py -noiseless tests/test_bell.in
```

**Run with realistic noise:**
```bash
python simulator.py -noise tests/test_bell.in -error 0.01
```

## Usage

### Basic Command

```bash
python simulator.py [MODE] <circuit_file> [OPTIONS]
```

### Modes (required - choose one)

- `-noiseless` - Perfect quantum simulation
- `-noise` - Realistic noisy simulation

### Optional Flags

- `-error <probability>` - Single-qubit gate error rate (default: 0.01)
- `-error2q <probability>` - Two-qubit gate error rate (default: 10x single-qubit)
- `-qubits <number>` - Override qubit count from circuit file
- `-readout <fidelity>` - Measurement accuracy 0-1 (default: 1.0 = perfect)

### Examples

**Deutsch-Jozsa algorithm:**
```bash
python simulator.py -noiseless tests/test_circuit.in
```

**Noisy quantum computer:**
```bash
python simulator.py -noise tests/test_ghz.in -error 0.02 -readout 0.95
```

**Performance testing:**
```bash
python simulator.py -noiseless tests/test_bell.in -qubits 25
```

**See all options:**
```bash
python simulator.py --help
```

## Circuit File Format

Create a `.in` file with this syntax:

```
circuit: N qubits

// Single-qubit gates
H(qubit_index)
X(qubit_index)

// Two-qubit gates
CNOT(control, target)

// Measurement
measure start..end
```

**Example - Bell State:**
```
circuit: 2 qubits
H(0)
CNOT(0,1)
measure 0..1
```

## Testing & Validation

**Validate correctness:**
```bash
python circuit_validator.py
```

**Run performance tests:**
```bash
python performance_test.py
```

## Included Circuits

### Test Circuits (`tests/`)
- `test_bell.in` - Bell state (quantum entanglement)
- `test_ghz.in` - GHZ state (3-qubit entanglement)
- `test_circuit.in` - Deutsch-Jozsa algorithm
- `test_superposition.in` - Hadamard superposition
- `test_x_gate.in`, `test_x_cancel.in`, `test_h_cancel.in` - Gate tests

### Benchmark Circuits (`circuits/`)
- `circuit1.in` - Deutsch-Jozsa balanced oracle
- `circuit2.in` - 11-qubit scalability test

## Project Structure

```
QuantumSim/
├── simulator.py              # Main simulator
├── performance_test.py       # Performance benchmarking
├── circuit_validator.py      # Correctness validation
├── circuits/                 # Benchmark circuits
└── tests/                    # Test examples
```

## Technical Details

For algorithm design, performance analysis, and optimization details, see the **project_report.ipynb**.

## Supported Gates

- **H (Hadamard)**: Creates superposition
- **X (Pauli-X)**: Quantum NOT gate
- **CNOT**: Controlled-NOT gate

## License

Open source - available for educational purposes.
