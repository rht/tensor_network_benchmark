import time

import numpy as np
from cuquantum import cutensornet as cutn

import common_tn
import circuits

np.random.seed(1)


# QAOA ansatz
from deqart_internal.circuit_converter import qiskit_to_cirq
import qsimcirq

np.random.seed(42)
simulator = qsimcirq.QSimSimulator()


def run_multiple_methods(
    circuit,
    qubits,
    index=0,
    enable_oe=False,
    enable_ctg=False,
    enable_cutn=False,
    enable_cusv=False,
):
    # pauli_string = {qubits[0]: "Z", qubits[1]: "Z"}
    pauli_string = {qubits[10]: "Z"}
    repeat_count = 2 if index == 0 else 1
    if enable_oe:
        print("Running with opt_einsum")
        # Run twice in the beginning to warm the GPU.
        for i in range(repeat_count):
            out = common_tn.run_with_oe(circuit, pauli_string)
        print("Output opt_einsum", out)

    if enable_ctg:
        print("Running with ctg")
        # Run twice in the beginning to warm the GPU.
        for i in range(repeat_count):
            out = common_tn.run_with_ctg(circuit, pauli_string)
        print("Output ctg", out)

    if enable_cutn:
        print("Running with cutn")
        out = common_tn.run_with_cutn(circuit, pauli_string)
        print("Output cutn", out)

    # cusv
    if enable_cusv:
        print("Running with cusv")
        circuit_cirq, _, _ = qiskit_to_cirq(circuit)
        monitor = common_tn.MemoryMonitor()
        monitor.start()
        tic = time.time()
        simulator.simulate(circuit_cirq)
        print("Elapsed cusv", round(time.time() - tic, 3))
        monitor.stop()


# import initialize_rqc
# run_multiple_methods(initialize_rqc.qc, initialize_rqc.qc.qubits)
# exit()

n_list = [22, 24, 30, 32]
for i, num_qubits in enumerate(n_list):
    print(num_qubits)
    ansatz, qubits = circuits.qaoa_ansatz_with_cost_included(num_qubits)
    # ansatz, qubits = circuits.make_vqe_QAOA_ansatz(num_qubits, high_entanglement=True)

    run_multiple_methods(ansatz, qubits, index=i, enable_cutn=True)
    print()

cutn.destroy(common_tn.handle)
