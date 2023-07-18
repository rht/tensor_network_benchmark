from collections import defaultdict

import numpy as np
from cuquantum import cutensornet as cutn

import common_tn
import circuits

np.random.seed(42)

# import initialize_rqc
# common_tn.run_multiple_methods(initialize_rqc.qc, initialize_rqc.qc.qubits)
# exit()


def run_exp(exp_name):
    full_output = defaultdict(list)
    n_list = None
    if exp_name == "vqe_realamplitudes_full":
        n_list = [12, 14, 20, 22]
    elif exp_name == "vqe_realamplitudes_linear":
        n_list = [12, 14, 20, 22]
    elif exp_name == "vqe_QAOA":
        n_list = [14, 18, 22, 26, 30]
    elif exp_name == "QPE":
        n_list = [2, 5, 10, 11]
    elif exp_name == "alexeev":
        n_list = [22, 24, 30, 32]

    for i, num_qubits in enumerate(n_list):
        print(num_qubits)
        if exp_name == "vqe_realamplitudes_full":
            ansatz, qubits = circuits.make_vqe_QAOA_ansatz(
                num_qubits, entanglement="full"
            )
        elif exp_name == "vqe_realamplitudes_linear":
            ansatz, qubits = circuits.make_vqe_QAOA_ansatz(
                num_qubits, entanglement="linear"
            )
        elif exp_name == "vqe_QAOA":
            ansatz, qubits = circuits.make_vqe_QAOA_ansatz(num_qubits, ansatz_type="TwoLocal")
        elif exp_name == "QPE":
            ansatz = circuits.make_QPE(num_qubits)
            qubits = ansatz.qubits
        elif exp_name == "alexeev":
            ansatz, qubits = circuits.qaoa_ansatz_with_cost_included(num_qubits)
        else:
            ansatz, qubits = circuits.qaoa_ansatz_with_cost_included(num_qubits)
        output = common_tn.run_multiple_methods(
            ansatz,
            qubits,
            index=i,
            enable_cutn=1,
            enable_cusv=0,
            enable_mps=0,
            enable_oe=0,
        )
        for k, v in output.items():
            full_output[k].append(v)
        print()
    full_output = {"elapsed": dict(full_output)}
    full_output["qubits"] = n_list
    print(full_output)


# run_exp("vqe_realamplitudes_full")
# run_exp("vqe_realamplitudes_linear")
# run_exp("vqe_QAOA")
# run_exp("QPE")
run_exp("alexeev")
cutn.destroy(common_tn.handle)
