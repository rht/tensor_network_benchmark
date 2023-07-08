import subprocess
import threading
import time

import cupy
from cuquantum import cutensornet as cutn
from cuquantum import contract, contract_path
from cuquantum import CircuitToEinsum


handle = cutn.create()
start_event = cupy.cuda.stream.Event()
stop_event = cupy.cuda.stream.Event()


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


def run_with_cutn(circuit, pauli_string):
    myconverter = CircuitToEinsum(circuit, backend=cupy)
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    # expression, operands = myconverter.amplitude(bitstring="0" * circuit.num_qubits)
    options = {
        "handle": handle,
        "blocking": "auto",
        "memory_limit": "80%",
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
    # optimize_options = None
    monitor = MemoryMonitor()
    monitor.start()
    start_event.record()
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
    monitor.stop()
    # In seconds
    elapsed = cupy.cuda.get_elapsed_time(start_event, stop_event) / 1e3
    print("Elapsed cutn", round(elapsed, 3))
    return output.get()


def run_with_oe(circuit, pauli_string):
    import opt_einsum as oe
    myconverter = CircuitToEinsum(circuit, backend=cupy)
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    monitor = MemoryMonitor()
    monitor.start()
    tic = time.time()
    # path, path_info = oe.contract_path(expression, *operands, memory_limit=10e9)
    memory_limit = "max_input"
    # memory_limit = None
    path, path_info = oe.contract_path(expression, *operands, memory_limit=memory_limit)
    print(path_info)
    output = oe.contract(
        expression, *operands, optimize=path, memory_limit=memory_limit
    )
    elapsed = time.time() - tic
    monitor.stop()
    print("Elapsed oe", round(elapsed, 3))
    return output


def run_with_ctg(circuit, pauli_string):
    import opt_einsum as oe
    import cotengra as ctg
    myconverter = CircuitToEinsum(circuit, backend=cupy)
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
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
    monitor.stop()
    print("Elapsed oe", round(elapsed, 3))
    return output


from qiskit import transpile
from qiskit_aer import AerSimulator

backend_mps_local = AerSimulator(method="matrix_product_state")


def run_circuit_mps(qc):
    backend = backend_mps_local
    shots = 4000
    # shots = 0
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

    print(result.get_counts())

    # sv = result.get_statevector()
    # pauli_op = PauliSumOp.from_list(["Z", 1.0])
    # expectation = pauli_op.expectation_value(Statevector(sv))
    # print(expectation)
    return time_taken_execute
