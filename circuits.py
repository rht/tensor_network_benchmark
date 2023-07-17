import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, RealAmplitudes

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


def make_vqe_QAOA_ansatz(num_qubits, high_entanglement=False):
    if high_entanglement:
        # Highly entangled ansatz
        ansatz = RealAmplitudes(num_qubits, reps=5, entanglement="full")
    else:
        # TwoLocal is for QAOA ansatz
        # ansatz = TwoLocal(num_qubits, "ry", "cz", reps=5, entanglement="linear")
        ansatz = TwoLocal(num_qubits, "rz", "cx", reps=5, entanglement="linear")

    ansatz = ansatz.bind_parameters(np.random.random(ansatz.num_parameters))
    qubits = ansatz.qubits
    return ansatz, qubits


# QPE
import qiskit.circuit.library as qcl
def prepare_A_circuit_for_qiskit(nqubits):
    from qiskit.opflow import X, Z

    xs = X
    zs = Z
    for i in range(nqubits - 1):
        xs = xs ^ X
        zs = zs ^ Z
    return xs.to_circuit()


def make_QPE(n):
    A = prepare_A_circuit_for_qiskit(n)
    pe = qcl.PhaseEstimation(n, A)
    return pe
