from dataclasses import dataclass


@dataclass
class Device(object):
    """
    device data
    """
    name: str
    compute_capability: str
    multiprocessors: int
    cuda_cores: int
    concurrent_threads: int
    gpu_clock_mhz: int
    gpu_memory_clock_mhz: int
    total_memory_mib: int
    free_memory_mib: int
