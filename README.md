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
- Noiseless: python simulator.py -noiseless <circuit_file>
- Noisy: python simulator.py -noise <circuit_file> -error <probability>

# Examples
### The below command runs the noiseless test_circuit.in
python simulator.py -noiseless test_circuit.in
### The command below runs the noisy test_bell.in file with a 0.01 error rate
python simulator.py -noise test_bell.in -error 0.01
### The command below runs the help command and shows more information about the inputs
python simulator.py --help

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

# The available test circuits:
- test_bell.in - Bell State
- test_ghz.in - GHZ State
- test_x_gate.in - X gate test
- test_superposition.in - Hadmard Superposition
- test_x_cancel.in - X gate self inverse test
- test_h_cancel.in - H gate self inverse test
- test_circuit.in - Deutsch-Jozsa algorithm