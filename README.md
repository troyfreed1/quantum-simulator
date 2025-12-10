# Quantum Circuit Simulator

A high-performance quantum circuit simulator supporting both noiseless and noisy simulation modes. Handles circuits with up to 30 qubits.

## Installation

```bash
# Clone the repository
git clone https://github.com/troyfreed1/quantum-simulator

# Install NumPy
pip3 install numpy
```
**Note:** All commands assume `python3` and `pip3`.


## Quick Start - Testing & Validation

**Run performance benchmarks:**
Measures performance of qubits showing the elapsed time and memory used

```bash
python3 performance_test.py
```

**Validate circuit correctness:**
Completes circuit example input files given to us in Canvas

```bash
python3 circuit_validator.py
```

**Run a simple example:**
```bash
python3 simulator.py -noiseless tests/test_bell.in
```

## Simulator Usage

**Basic command:**
```bash
python3 simulator.py [MODE] <circuit_file> [OPTIONS]
```

**Modes (required - choose one):**
- `-noiseless` - Perfect quantum simulation (no errors)
- `-noise` - Realistic noisy simulation (includes gate errors)

**Optional Arguments:**
- `-error <probability>` - Single-qubit gate error rate (default: 0.01, range: 0.0-1.0)
- `-error2q <probability>` - Two-qubit gate error rate (default: 10x single-qubit rate)
- `-qubits <number>` - Override number of qubits from circuit file (for performance testing)
- `-readout <fidelity>` - Measurement accuracy from 0.0-1.0 (default: 1.0 = perfect)

**Examples:**
```bash
# Bell state
python3 simulator.py -noiseless tests/test_bell.in

# GHZ state with noisy simulation (1% error rate)
python3 simulator.py -noise tests/test_ghz.in -error 0.01

# DJ algorithm from Canvas benchmarks
python3 simulator.py -noiseless circuits/circuit1.in

# Scalability test (11-qubit circuit)
python3 simulator.py -noiseless circuits/circuit2.in

# Noisy simulation with custom 2-qubit error rate
python3 simulator.py -noise tests/test_circuit.in -error 0.01 -error2q 0.05

# Superposition test with imperfect readout (95% accuracy)
python3 simulator.py -noise tests/test_superposition.in -error 0.02 -readout 0.95

# Performance testing with 25 qubits
python3 simulator.py -noiseless tests/test_bell.in -qubits 25
```

## Example Input/Output

**Example circuit file (tests/test_bell.in):**
```
circuit: 2 qubits
H(0)
CNOT(0,1)
measure 0..1
```

**Example output:**
```
Loading circuit: tests/test_bell.in
Circuit has 2 qubits
Measure qubits: [0, 1]

Circuit: 2 qubits, 2 gates
Mode: Noiseless

Simulation
Initialized 2-qubit system (state vector: 4)

Applying gates...
Applied 2 gates


Final State

 Current Quantum State:
|00> : 0.7071+0.0000j
|11> : 0.7071+0.0000j

Measurement
Measuring qubits [0, 1] (1000 shots)

Results Summary
|00> :  496 █████████████████████████████████████████████████
|11> :  504 ██████████████████████████████████████████████████
```

**Understanding the output:**

- **Final State**: Shows quantum amplitudes for each basis state. The amplitude 0.7071 = 1/√2, meaning each state has 50% probability

- **Measurement Results**: Shows outcomes from 1000 measurement shots
  - `|00>: 496` means the state |00⟩ was measured 496 times (~49.6%)
  - `|11>: 504` means the state |11⟩ was measured 504 times (~50.4%)
  - Bar chart visualizes the distribution (each █ = 10 measurements)

- **Note**: Small deviations from 50/50 (like 496/504) are from the statistical variation, not errors

## Technical Details

For any other information about our algorithm design, performance analysis, and optimization details, visit **project_report.ipynb**.
