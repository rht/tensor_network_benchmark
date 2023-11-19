import cupy
from cuquantum import CircuitToEinsum
from cuquantum import cutensornet as cutn
from cuquantum import contract_path

import circuits

handle = cutn.create()


def calculate_entanglement_entropy_and_tn_cost(circuit, qubits):
    from qiskit.quantum_info import entropy, partial_trace
    from qiskit import execute
    from qiskit_aer import Aer

    simulator = Aer.get_backend("statevector_simulator")
    job = execute(circuit, simulator)
    result = job.result()
    sv = result.get_statevector(circuit)
    # Partition the system and calculate the reduced density matrix
    # For instance, tracing out the last two qubits
    reduced_matrix = partial_trace(sv, [len(qubits) - 2, len(qubits) - 1])
    ee = entropy(reduced_matrix, base=2)

    myconverter = CircuitToEinsum(circuit, backend=cupy)
    idx = 10
    pauli_string = {qubits[idx]: "Z"}
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)

    optimize_options = {"seed": 1, "samples": 10}
    options = {
        "handle": handle,
        "blocking": "auto",
    }
    path, info = contract_path(
        expression, *operands, options=options, optimize=optimize_options
    )
    return ee, info.opt_cost


def main():
    entropies = []
    costs = []
    for num_qubits in range(15, 25):
        print("Doing", num_qubits)
        circuit, qubits = circuits.make_vqe_QAOA_ansatz(
            num_qubits, entanglement="linear"
        )
        ee, cost = calculate_entanglement_entropy_and_tn_cost(circuit, qubits)
        entropies.append(ee)
        costs.append(cost)
    print({"label": "QAOA", "entropy": entropies, "opt_cost": costs})


main()
