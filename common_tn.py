import subprocess
import threading
import time

import cupy
import qsimcirq
from cuquantum import cutensornet as cutn
from cuquantum import contract, contract_path
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit, ClassicalRegister

from deqart_internal.circuit_converter import qiskit_to_cirq


simulator = qsimcirq.QSimSimulator()
handle = cutn.create()
start_event = cupy.cuda.stream.Event()
stop_event = cupy.cuda.stream.Event()
mempool = cupy.get_default_memory_pool()


def get_cupy_memory():
    return cupy.cuda.Device().mem_info[0]


def get_gpu_memory():
    def _output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    # MB
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return round(memory_used_values[0] / 1e3, 2)


class MemoryMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_memory = 0
        cupy.get_default_memory_pool().free_all_blocks()

    def monitor_memory(self):
        while not self.stop_event.is_set():
            current_memory = get_gpu_memory()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(self.interval)

    def start(self):
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()
        print("Peak memory usage: {} gigabytes".format(self.peak_memory))
        return self.peak_memory


def run_with_cutn(circuit, exp_ops):
    # See https://docs.nvidia.com/cuda/cuquantum/latest/python/api/generated/cuquantum.contract_path.html?highlight=contract_path#cuquantum.contract_path
    # https://docs.nvidia.com/cuda/cuquantum/latest/python/api/generated/cuquantum.contract.html#cuquantum.contract
    expression, operands = exp_ops
    # expression, operands = myconverter.amplitude(bitstring="0" * circuit.num_qubits)
    available_memory = get_cupy_memory()
    print("Available_memory", round(available_memory / (1024**3), 3), "GB")
    options = {
        "handle": handle,
        "blocking": "auto",
        "memory_limit": 0.8 * available_memory,
        # "memory_limit": "80%",
    }
    optimize_options = {
        "samples": 5,  # default is 0 (disabled)
        "slicing": {
            "disable_slicing": 0,
            "memory_model": 1,  # 0 is heuristic, 1 is cutensor (default)
            "min_slices": 10,  # default is 1
            "slice_factor": 2,  # default is 32
        },
        "cost_function": 0,  # 0 for FLOPS (default), 1 for time
        "reconfiguration": {
            "num_iterations": 800,  # default is 500; good values are within 500-1000
            # Higher number means more time spent in reconfiguration, scales exponentially
            "num_leaves": 10,  # default is 8
        },
    }
    # It is recommended to use vanilla optimize options as much as possible.
    # Seed it for deterministic result
    seed = 1
    monitor = MemoryMonitor()
    monitor.start()
    start_event.record()
    if False:
        # If we combine path finding and contraction into 1 function call.
        output = contract(
            expression,
            *operands,
            optimize={"seed": seed},
            options=options,
        )
    else:
        # Separate path finding and contraction step
        optimize_options = {"seed": seed}
        path, info = contract_path(
            expression, *operands, options=options, optimize=optimize_options
        )

        output = contract(
            expression,
            *operands,
            optimize={"path": path, "slicing": info.slices},
            options=options,
        )
    stop_event.record()
    stop_event.synchronize()
    peak_memory = monitor.stop()
    # In seconds
    elapsed = cupy.cuda.get_elapsed_time(start_event, stop_event) / 1e3
    elapsed = round(elapsed, 3)
    print("Elapsed cutn", elapsed)
    output_cpu = output.get()
    del output
    return output_cpu, elapsed, peak_memory


def run_with_oe(circuit, exp_ops):
    import opt_einsum as oe

    expression, operands = exp_ops
    monitor = MemoryMonitor()
    monitor.start()
    tic = time.time()
    # path, path_info = oe.contract_path(expression, *operands, memory_limit=10e9)
    memory_limit = "max_input"
    memory_limit = None
    if True:
        output = oe.contract(expression, *operands, memory_limit=memory_limit)
    else:
        # Separate path finding and contraction
        path, path_info = oe.contract_path(
            expression, *operands, memory_limit=memory_limit
        )
        output = oe.contract(
            expression, *operands, optimize=path, memory_limit=memory_limit
        )
    elapsed = time.time() - tic
    peak_memory = monitor.stop()
    elapsed = round(elapsed, 3)
    print("Elapsed oe", elapsed)
    return output, elapsed, peak_memory


def run_with_ctg(circuit, exp_ops):
    import opt_einsum as oe
    import cotengra as ctg

    expression, operands = exp_ops
    monitor = MemoryMonitor()
    monitor.start()
    opt = ctg.HyperOptimizer(
        # minimize="combo",
        reconf_opts={},
        # # if you did need dynamic slicing:
        # 1 GiB
        slicing_reconf_opts=dict(target_size=1 * 1024**3),
        progbar=True,
    )
    tic = time.time()
    path, path_info = oe.contract_path(expression, *operands, optimize=opt)
    output = oe.contract(expression, *operands, optimize=path)
    elapsed = time.time() - tic
    peak_memory = monitor.stop()
    elapsed = round(elapsed, 3)
    print("Elapsed oe", elapsed)
    return output, elapsed, peak_memory


from qiskit import transpile
from qiskit_aer import AerSimulator

backend_mps_local = AerSimulator(method="matrix_product_state")


def run_circuit_mps(qc):
    backend = backend_mps_local
    shots = 4000
    # shots = 0
    # Sometimes, qct.measure_all seems to be much faster than measuring just
    # 1 qubit. TODO investigate why.
    qct = transpile(qc, backend)
    print(
        f"\ngoing for qubits: {len(qc.qubits)}\t gates: {len(qct.data)} depth: {qct.depth()}\t "
    )
    result = backend.run(qct, shots=shots).result()
    print(
        f"qubits: {qc.num_qubits}\t gates: {len(qct.data)}\t depth: {qct.depth()}\t "
        f"t: {result.time_taken}\t success: {result.success}\t status: {result.status}"
    )
    time_taken_execute = result._metadata["metadata"]["time_taken_execute"]
    print("time taken execute", time_taken_execute)
    # These are not actual memory used. Those are just settings
    # print("max memory mb", result._metadata["metadata"]["max_memory_mb"])
    # print("max gpu memory mb", result._metadata["metadata"]["max_gpu_memory_mb"])

    # print(result.get_counts())

    # sv = result.get_statevector()
    # pauli_op = PauliSumOp.from_list(["Z", 1.0])
    # expectation = pauli_op.expectation_value(Statevector(sv))
    # print(expectation)
    return time_taken_execute


def run_with_mps(qc):
    # run_circuit_mps but with memory monitor
    monitor = MemoryMonitor()
    monitor.start()
    tic = time.time()
    run_circuit_mps(qc)
    elapsed = round(time.time() - tic, 3)
    print("Elapsed MPS", elapsed)
    peak_memory = monitor.stop()
    return elapsed, peak_memory


def run_with_cusv(circuit):
    circuit_cirq, _, _ = qiskit_to_cirq(circuit)
    monitor = MemoryMonitor()
    monitor.start()
    tic = time.time()
    simulator.simulate(circuit_cirq)
    elapsed = round(time.time() - tic, 3)
    print("Elapsed cusv", elapsed)
    peak_memory = monitor.stop()
    return elapsed, peak_memory


def run_multiple_methods(
    circuit,
    qubits,
    index=0,
    enable_oe=False,
    enable_ctg=False,
    enable_cutn=False,
    enable_cusv=False,
    enable_mps=False,
    mps_measure_1qubit=True,
    mode=None,
):
    if mode is None:
        mode = "expectation_pauli_1"
    output = {}
    output_memory = {}
    myconverter = CircuitToEinsum(circuit, backend=cupy)

    if mode == "expectation_pauli_1":
        idx = 10 if len(qubits) > 10 else 0
        pauli_string = {qubits[idx]: "Z"}
        exp_ops = myconverter.expectation(pauli_string, lightcone=True)
    elif mode == "expectation_pauli_2":
        idx = 10 if len(qubits) > 10 else 0
        pauli_string = {qubits[idx]: "Z", qubits[1]: "Z"}
        exp_ops = myconverter.expectation(pauli_string, lightcone=True)
    elif mode == "bitstring_1":
        # Only 1 bitstring
        bitstring = "0" * len(qubits)
        exp_ops = myconverter.amplitude(bitstring=bitstring)
    else:
        raise Exception(f"Unknown mode {mode}")

    repeat_count = 2 if index == 0 else 1
    if enable_oe:
        print("Running with opt_einsum")
        # Run twice in the beginning to warm the GPU.
        for i in range(repeat_count):
            out, elapsed, peak_memory = run_with_oe(circuit, exp_ops)
        print("Output opt_einsum", out)
        # Important: We must free the GPU memory allocation
        del out
        mempool.free_all_blocks()
        output["opt_einsum"] = elapsed
        output_memory["opt_einsum"] = peak_memory

    if enable_ctg:
        print("Running with ctg")
        # Run twice in the beginning to warm the GPU.
        for i in range(repeat_count):
            out, elapsed, peak_memory = run_with_ctg(circuit, exp_ops)
        print("Output ctg", out)
        # Important: We must free the GPU memory allocation
        del out
        mempool.free_all_blocks()
        output["ctg"] = elapsed
        output_memory["ctg"] = peak_memory

    if enable_cutn:
        print("Running with cutn")
        out, elapsed, peak_memory = run_with_cutn(circuit, exp_ops)
        print("Output cutn", out)
        # Important: We must free the GPU memory allocation
        del out
        mempool.free_all_blocks()
        output["cutn"] = elapsed
        output_memory["cutn"] = peak_memory

    # cusv
    if enable_cusv:
        print("Running with cusv")
        elapsed, peak_memory = run_with_cusv(circuit)
        # Important: We must free the GPU memory allocation
        mempool.free_all_blocks()
        output["cusv"] = elapsed
        output_memory["cusv"] = peak_memory

    if enable_mps:
        if mps_measure_1qubit:
            cr = ClassicalRegister(1)
            circuit.add_register(cr)
            circuit.measure(idx, 0)
        else:
            circuit.measure_all()
        elapsed, peak_memory = run_with_mps(circuit)
        # Important: We must free the GPU memory allocation
        mempool.free_all_blocks()
        output["mps"] = elapsed
        output_memory["mps"] = peak_memory
    return output, output_memory
