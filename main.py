import time

import networkx as nx
import numpy as np
from cuquantum import cutensornet as cutn
from qiskit import QuantumCircuit

import common_tn

np.random.seed(1)


# QAOA ansatz
from qiskit.circuit.library import TwoLocal
import numpy as np
from deqart_internal.circuit_converter import qiskit_to_cirq
import qsimcirq

np.random.seed(42)
simulator = qsimcirq.QSimSimulator()


def qaoa_ansatz(num_qubits, reps):
    qc = QuantumCircuit(num_qubits)
    qubits = qc.qubits
    for i in range(reps):
        for j in range(num_qubits):
            qc.rz(np.random.random(), qubits[j])
        for j in range(num_qubits - 1):
            qc.cx(qubits[j], qubits[j + 1])
    return qc, qubits


def qaoa_ansatz_with_cost_included(num_qubits):
    # This is the ansatz found in
    # https://arxiv.org/abs/2012.02430 for the MaxCut
    # problem.
    print("QAOA ansatz with cost included")
    degree = 3
    n = num_qubits
    G = nx.random_regular_graph(degree, n)
    qc = QuantumCircuit(num_qubits, 1)
    qc.h(range(num_qubits))
    repetitions = 2
    for _ in range(repetitions):
        # Cost Hamiltonian
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp == 0:
                    continue
                # TODO fix for weighted graph!
                # w[i, j] = temp["weight"]
                qc.rzz(np.random.random(), i, j)
        # Mixer
        for i in range(n):
            beta = np.random.random()
            qc.rx(beta, i)
    return qc, qc.qubits


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

def make_vqe_QAOA_ansatz(num_qubits):
    # TwoLocal is for QAOA ansatz
    # ansatz = TwoLocal(num_qubits, "ry", "cz", reps=5, entanglement="linear")
    ansatz = TwoLocal(num_qubits, "rz", "cx", reps=5, entanglement="linear")
    # Highly entangled ansatz
    # ansatz = RealAmplitudes(num_qubits, reps=5, entanglement="full")
    ansatz = ansatz.bind_parameters(np.random.random(ansatz.num_parameters))
    qubits = ansatz.qubits
    return ansatz, qubits

n_list = [22, 24, 30, 32]
for i, num_qubits in enumerate(n_list):
    print(num_qubits)
    ansatz, qubits = qaoa_ansatz_with_cost_included(num_qubits)

    run_multiple_methods(ansatz, qubits, index=i, enable_cutn=True)
    print()

# QPE
# import qiskit.circuit.library as qcl
# def prepare_A_circuit_for_qiskit(nqubits):
#    from qiskit.opflow import X, Z, PauliTrotterEvolution
#
#    xs = X
#    zs = Z
#    for i in range(nqubits - 1):
#        xs = xs ^ X
#        zs = zs ^ Z
#    return xs.to_circuit()
#
# ns = [11, 12, 13 ,14]
# for n in ns:
#    print(n)
#    A = prepare_A_circuit_for_qiskit(n)
#    pe = qcl.PhaseEstimation(n, A)
#    print("Running with cutn")
#    for i in range(1):
#        out = run_with_cutn(pe, {pe.qubits[0]: "Z"})
#    print("Output cutn", out)
#    print("Running with opt_einsum")
#    for i in range(2):
#        out = run_with_oe(pe, {pe.qubits[0]: "Z"})
cutn.destroy(common_tn.handle)
