# See https://github.com/youngseok-kim1/Evidence-for-the-utility-of-quantum-computing-before-fault-tolerance
from collections import defaultdict

import numpy as np
import seaborn as sns
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import YGate
from qiskit.quantum_info import Pauli
from qiskit_aer import Aer

import common_tn

simulator = Aer.get_backend("statevector_simulator")
np.random.seed(1)


def get_statevector(qc):
    job = execute(qc, simulator)
    result = job.result()
    sv = result.get_statevector(qc)
    return sv


def make_circuit_12(theta_h, G=None):
    num_qubits = 12
    qc = QuantumCircuit(num_qubits)
    lambdas = [np.random.random() for _ in range(num_qubits)]

    # 0-1-2-3--4
    # |        |
    # 11       5
    # |        |
    # 10-9-8-7-6

    def get_probability(_lambda):
        return (1 - np.exp(-2 * (G - 1) * _lambda)) / 2

    def zz(i, j):
        qc.sdg(i)
        qc.sdg(j)
        sy = YGate().power(1 / 2)

        def maybe_twirl(idx, create):
            if not (np.random.random() <= get_probability(lambdas[idx])):
                return
            if create:
                qc.sdg(idx)
            else:
                qc.s(idx)
            qc.h(idx)

        qc.append(sy, [j])
        if G is not None:
            maybe_twirl(i, False)
            maybe_twirl(j, False)
            qc.cx(i, j)
            maybe_twirl(i, True)
            maybe_twirl(j, True)
        else:
            qc.cx(i, j)
        qc.append(sy, [j])

    for i in range(num_qubits):
        qc.rx(theta_h, i)

    def do_trotter_step():
        # red
        zz(1, 2)
        zz(5, 6)
        zz(7, 8)
        zz(11, 0)
        # blue
        zz(2, 3)
        zz(4, 5)
        zz(6, 7)
        zz(8, 9)
        zz(10, 11)
        # green
        zz(0, 1)
        zz(3, 4)
        zz(9, 10)

    for i in range(4):
        do_trotter_step()
    return qc


def make_circuit_27(theta_h):
    num_qubits = 27
    qc = QuantumCircuit(num_qubits)

    #     13 14-15-16-17-18
    #     |        |
    #     11       12
    #     |        |
    # 0-1-2-3-4-5--6-7-8-9-10
    #         |        |
    #         19       20
    #         |        |
    #  21-22-23-24-25  26

    #     13 14r15g16r17b18
    #     b        b
    #     11       12
    #     r        g
    # 0r1b2g3r4g5-r6b7g8b9g10
    #         b        r
    #         19       20
    #         g        b
    #  21g22b23r24b25  26

    def zz(i, j):
        qc.sdg(i)
        qc.sdg(j)
        sy = YGate().power(1 / 2)

        qc.append(sy, [j])
        qc.cx(i, j)
        qc.append(sy, [j])

    for i in range(num_qubits):
        qc.rx(theta_h, i)

    def do_trotter_step():
        # red
        zz(14, 15)
        zz(16, 17)
        zz(11, 2)
        zz(0, 1)
        zz(3, 4)
        zz(5, 6)
        zz(8, 20)
        zz(23, 24)
        # blue
        zz(17, 18)
        zz(13, 11)
        zz(16, 12)
        zz(1, 2)
        zz(6, 7)
        zz(8, 9)
        zz(4, 19)
        zz(20, 26)
        zz(22, 23)
        zz(24, 25)
        # green
        zz(15, 16)
        zz(12, 6)
        zz(2, 3)
        zz(4, 5)
        zz(7, 8)
        zz(9, 10)
        zz(19, 23)
        zz(21, 22)

    for i in range(2):
        do_trotter_step()
    return qc


def make_circuit_127(theta_h):
    num_qubits = 127
    qc = QuantumCircuit(num_qubits)

    def zz(i, j):
        qc.sdg(i)
        qc.sdg(j)
        sy = YGate().power(1 / 2)

        qc.append(sy, [j])
        qc.cx(i, j)
        qc.append(sy, [j])

    for i in range(num_qubits):
        qc.rx(theta_h, i)

    def do_trotter_step():
        # red (48)
        red = [
            (1, 2),
            (4, 5),
            (6, 7),
            (8, 9),
            (12, 13),
            (0, 14),
            (15, 22),
            (17, 30),
            (20, 21),
            (23, 24),
            (25, 26),
            (31, 32),
            (28, 35),
            (33, 39),
            (36, 51),
            (43, 44),
            (45, 46),
            (47, 48),
            (49, 50),
            (37, 52),
            (41, 53),
            (55, 68),
            (56, 57),
            (59, 60),
            (61, 62),
            (63, 64),
            (66, 67),
            (58, 71),
            (70, 74),
            (72, 81),
            (73, 85),
            (76, 77),
            (78, 79),
            (83, 84),
            (87, 88),
            (92, 102),
            (94, 95),
            (96, 97),
            (98, 99),
            (103, 104),
            (105, 106),
            (107, 108),
            (100, 110),
            (109, 114),
            (115, 116),
            (118, 119),
            (120, 121),
            (122, 123),
        ]
        for i, j in red:
            zz(i, j)

        # blue
        blue = [
            (2, 3),
            (5, 6),
            (9, 10),
            (11, 12),
            (4, 15),
            (8, 16),
            (14, 18),
            (19, 20),
            (21, 22),
            (26, 27),
            (28, 29),
            (30, 31),
            (24, 34),
            (35, 47),
            (38, 39),
            (40, 41),
            (42, 43),
            (44, 45),
            (48, 49),
            (53, 60),
            (54, 64),
            (57, 58),
            (62, 63),
            (65, 66),
            (67, 68),
            (69, 70),
            (71, 77),
            (74, 89),
            (75, 76),
            (79, 80),
            (81, 82),
            (84, 85),
            (86, 87),
            (83, 92),
            (90, 94),
            (91, 98),
            (93, 106),
            (99, 100),
            (101, 102),
            (96, 109),
            (104, 111),
            (108, 112),
            (110, 118),
            (113, 114),
            (116, 117),
            (121, 122),
            (123, 124),
            (125, 126)
        ]
        for i, j in blue:
            zz(i, j)
        # green
        green = [
            (0, 1),
            (3, 4),
            (7, 8),
            (10, 11),
            (12, 17),
            (16, 26),
            (18, 19),
            (22, 23),
            (24, 25),
            (27, 28),
            (29, 30),
            (20, 33),
            (32, 36),
            (34, 43),
            (37, 38),
            (39, 40),
            (41, 42),
            (46, 47),
            (50, 51),
            (45, 54),
            (49, 55),
            (52, 56),
            (58, 59),
            (60, 61),
            (64, 65),
            (68, 69),
            (62, 72),
            (66, 73),
            (77, 78),
            (80, 81),
            (82, 83),
            (85, 86),
            (88, 89),
            (75, 90),
            (79, 91),
            (87, 93),
            (95, 96),
            (97, 98),
            (100, 101),
            (102, 103),
            (104, 105),
            (106, 107),
            (111, 122),
            (112, 126),
            (114, 115),
            (117, 118),
            (119, 120),
            (124, 125),
        ]
        for i, j in green:
            zz(i, j)

    for i in range(2):
        do_trotter_step()
    return qc


three_eight_pi = 3 * np.pi / 8
theta_hs = np.concatenate(
    (
        np.linspace(0, np.pi / 8, 4),
        np.arange(np.pi / 8 + 0.1, three_eight_pi, 0.2),
        np.linspace(three_eight_pi, np.pi / 2, 5),
    )
)

# TODO what is this?
# arr = []
# for G in np.linspace(1, 2, 10):
#     qc = make_circuit_12(three_eight_pi, G=G)
#     sv = get_statevector(qc)
#     i_str = "I" * 12
#     idx = 10
#     pauli_str = i_str[:idx] + "Z" + i_str[idx + 1 :]
#     z_idx = sv.expectation_value(Pauli(pauli_str))
#     arr.append(z_idx)
#     print(z_idx)
# g = sns.scatterplot(arr)
# g.figure.savefig("ttemp.png")
# exit()

if 1:
    full_output = defaultdict(list)
    full_output_memory = defaultdict(list)
    for i, theta_h in enumerate(theta_hs):
        qc = make_circuit_127(theta_h)
        output, output_memory = common_tn.run_multiple_methods(
            qc,
            qc.qubits,
            index=i,
            enable_cutn=1,
            enable_cusv=0,
            enable_mps=1,
            enable_oe=1,
            mps_measure_1qubit=True,
        )
        for k, v in output.items():
            full_output[k].append(v)
        for k, v in output_memory.items():
            full_output_memory[k].append(v)
    full_output = {"elapsed": dict(full_output), "memory": dict(full_output_memory)}
    full_output["theta_hs"] = theta_hs
    print(full_output)


if False:
    mzs = []
    for theta_h in theta_hs:
        qc = make_circuit_12(theta_h)
        sv = get_statevector(qc)

        # Magnetization in the z direction
        mz = sv.expectation_value(Pauli("Z" * 12))
        print(theta_h, mz)

        # Another way to compute mz
        # probs = sv.probabilities_dict()
        # mz = sum(v * (-1) ** (k.count("1")) for k, v in probs.items())

        mzs.append(mz)

    # Figure 3
    # g = sns.lineplot(x=theta_hs, y=mzs)
    g = sns.scatterplot(x=theta_hs, y=mzs)
    g.set(
        xlabel="$R_x$ angle $\\theta_h$",
        ylabel="Magnetization $M_z$",
    )
    g.set_xticks(np.array([0, 0.125, 0.25, 0.375, 0.5]) * np.pi)
    g.set_xticklabels(["0", r"$\pi$/8", r"$\pi$/4", r"$3\pi$/8", r"$\pi$/2"])
    g.figure.savefig("magnetization_z.png")
