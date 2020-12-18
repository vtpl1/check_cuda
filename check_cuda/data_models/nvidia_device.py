from dataclasses import dataclass, field
from typing import List


@dataclass
class NvidiaDevice:
    '''
    Nvidia device data
    '''
    gpu_id: int
    name: str
    compute_capability_major: int
    compute_capability_minor: int
    cores: int
    concurrent_threads: int
    gpu_clock_mhz: int
    memory_clock_mhz: int
    total_memory_mib: int
    free_memory_mib: int
