"""
Validation script for benchmark circuits
Tests that circuit outputs match expected results
"""

import json
import sys
from simulator import run_simulation

def validate_circuit(circuit_file, expected_file):
    """
    Validate that circuit produces expected output distribution

    Args:
        circuit_file: Path to .in file
        expected_file: Path to .out file with expected distribution

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Validating: {circuit_file}")
    print(f"{'='*70}")

    # Load expected results
    with open(expected_file, 'r') as f:
        expected = json.load(f)

    print(f"Expected output: {expected}")

    # Run simulation
    sim, results = run_simulation(circuit_file, noise_mode=False)

    if results is None:
        print("FAILED: No measurement results returned")
        return False

    # Move to probability distribution
    total_shots = sum(results.values())
    measured_dist = {state: count/total_shots for state, count in results.items()}

    print(f"\nMeasured distribution (1000 shots):")
    for state, prob in sorted(measured_dist.items()):
        print(f"  |{state}⟩: {prob:.3f}")

    # check all expected states are present with reasonable probability
    tolerance = 0.15  # Allowable deviation

    passed = True
    for state, expected_prob in expected.items():
        measured_prob = measured_dist.get(state, 0.0)
        diff = abs(measured_prob - expected_prob)

        if diff > tolerance:
            print(f" FAILED for state |{state}⟩:")
            print(f"   Expected: {expected_prob:.3f}")
            print(f"   Measured: {measured_prob:.3f}")
            print(f"   Difference: {diff:.3f} (tolerance: {tolerance})")
            passed = False
        else:
            print(f"✓ State |{state}⟩: {measured_prob:.3f} (expected {expected_prob:.3f}, diff={diff:.3f})")

    # Check for unexpected states with significant probability
    for state, prob in measured_dist.items():
        if state not in expected and prob > 0.05:  # >5% probability for unexpected state
            print(f"\n  WARNING: Unexpected state |{state}⟩ with {prob:.3f} probability")

    if passed:
        print(f"\n PASSED: {circuit_file}")
    else:
        print(f"\n FAILED: {circuit_file}")

    return passed


if __name__ == "__main__":
    test_cases = [
        ('circuit1.in', 'circuit1.out'),
        ('circuit2.in', 'circuit2.out'),
    ]

    print("="*70)
    print("BENCHMARK CIRCUIT VALIDATION")
    print("="*70)

    results = []
    for circuit_in, circuit_out in test_cases:
        try:
            passed = validate_circuit(circuit_in, circuit_out)
            results.append((circuit_in, passed))
        except FileNotFoundError as e:
            print(f"\n File not found: {e}")
            results.append((circuit_in, False))
        except Exception as e:
            print(f"\n Error: {e}")
            results.append((circuit_in, False))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for circuit, passed in results:
        status = " PASSED" if passed else " FAILED"
        print(f"{status}: {circuit}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n All benchmark circuits validated successfully!")
        sys.exit(0)
    else:
        print(f"\n  {total_count - passed_count} test(s) failed")
        sys.exit(1)
