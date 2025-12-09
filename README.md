# quantum-simulator
Quantum computing circuit simulator project

# Quantum Circuit Simulator

A quantum circuit simulator that takes an input file and outputs data related to how the inputted algorithm performs and runs

## Installation
- Python 3.7+
- Numpy

## Setup
CD into the directory after cloning the github. Then create a virtual environment and install Numpy

# Usage
- Noiseless: `python simulator.py -noiseless <circuit_file>`
- Noisy: `python simulator.py -noise <circuit_file> -error <probability>`
- Custom Qubits: `python simulator.py -noiseless <circuit_file> -qubits <number>`

## Optional Flags
- `-qubits N`: Override the number of qubits specified in the circuit file (useful for testing performance scaling)

# Examples
### Run noiseless simulation
```bash
python simulator.py -noiseless test_circuit.in
```

### Run noisy simulation with 1% error rate
```bash
python simulator.py -noise test_bell.in -error 0.01
```

### Run with custom qubit count (for performance testing)
```bash
python simulator.py -noiseless test_bell.in -qubits 10
```

### Show help and all available options
```bash
python simulator.py --help
```

# Circuit Input Format

```
// Comments are made like this //

circuit: N qubits

//Single Qubit Gates
X(qubit_index)
H(qubit_index)

//Two-qubit Gates
CNOT(control, target)

// Measurement
measure start..end
```

## Example
```
// Bell state circuit
circuit: 2 qubits
H(0)
CNOT(0,1)
measure 0..1
```

# Input and Output Examples

Input test_bell.in
```
circuit: 2 qubits
H(0)
CNOT(0,1)
measure 0..1
```

Output Noiseless:
```
Loading circuit: test_bell.in
Circuit has 2 qubits
Added gate: H on qubit 0
Added gate: CNOT on qubits 0, 1
Measure qubits: [0, 1]

Circuit Summary:
 Qubits: 2
 Gates: 2
 Measurements: [0, 1]
 Mode: Noiseless


Simulation
Initialized 2-qubit system
State vector size: 4
Initial state: |00>

Applying gates...

[Gate 1]/2
 Current Quantum State:
|00> : 0.7071+0.0000j
|10> : 0.7071+0.0000j

[Gate 2]/2
 Current Quantum State:
|00> : 0.7071+0.0000j
|11> : 0.7071+0.0000j


Final Gate

 Current Quantum State:
|00> : 0.7071+0.0000j
|11> : 0.7071+0.0000j
Measurement
Measuring qubits [0, 1] (1000 shots)

Measurement Results:
|00> : 500/1000 = 50.0%
|11> : 500/1000 = 50.0%


Results Summary
|00> :  500 ██████████████████████████████████████████████████
|11> :  500 ██████████████████████████████████████████████████
```

Output Noisy with 0.01 error probability:
```
Loading circuit: test_bell.in
Circuit has 2 qubits
Added gate: H on qubit 0
Added gate: CNOT on qubits 0, 1
Measure qubits: [0, 1]

Circuit Summary:
 Qubits: 2
 Gates: 2
 Measurements: [0, 1]

 Mode: Noise (error rate = 0.01)


Simulation
Initialized 2-qubit system
State vector size: 4
Initial state: |00>

Applying gates...

[Gate 1]/2
 Current Quantum State:
|00> : 0.7071+0.0000j
|10> : 0.7071+0.0000j

[Gate 2]/2
 Current Quantum State:
|00> : 0.7071+0.0000j
|11> : 0.7071+0.0000j


Final Gate

 Current Quantum State:
|00> : 0.7071+0.0000j
|11> : 0.7071+0.0000j
Measurement
Measuring qubits [0, 1] (1000 shots)

Measurement Results:
|00> : 485/1000 = 48.5%
|11> : 515/1000 = 51.5%


Results Summary
|00> :  485 ████████████████████████████████████████████████
|11> :  515 ███████████████████████████████████████████████████
```

# Performance

## Scalability

The simulator efficiently handles quantum circuits of varying sizes:

| Qubits | State Vector Size | Memory (complex64) | Typical Runtime |
|--------|------------------|-------------------|-----------------|
| 10 | 1,024 | 8 KB | <0.003s |
| 15 | 32,768 | 256 KB | ~0.007s |
| 20 | 1,048,576 | 8 MB | ~0.25s |
| 25 | 33,554,432 | 256 MB | ~9.4s |
| 27 | 134,217,728 | 1 GB | ~9.9s |
| 28 | 268,435,456 | 2 GB | ~26s |
| 29 | 536,870,912 | 4 GB | ~78s |
| 30 | 1,073,741,824 | 8 GB | ~163s |

## Performance Optimizations

The simulator includes multiple key optimizations:

1. **Complex64 Data Type**: 50% memory reduction compared to complex128
2. **Vectorized CNOT Gate**: Uses NumPy bitwise operations instead of Python loops (~400-5000x faster)
3. **Gate Matrix Caching**: Pre-computes gate matrices to avoid repeated allocation (~2-5x faster)
4. **Gate Cancellation**: Automatically removes self-inverse gate pairs in noiseless mode
5. **Fast RNG (PCG64)**: 10-20% faster random number generation
6. **Value Caching**: Reduced allocations in hot paths for better performance

These optimizations enable simulation of **up to 30 qubits** on systems with 8GB+ RAM.

# The available test circuits:
- test_bell.in - Bell State
- test_ghz.in - GHZ State
- test_x_gate.in - X gate test
- test_superposition.in - Hadmard Superposition
- test_x_cancel.in - X gate self inverse test
- test_h_cancel.in - H gate self inverse test
- test_circuit.in - Deutsch-Jozsa algorithm