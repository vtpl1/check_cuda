from dataclasses import dataclass, field
from typing import List


@dataclass
class Device():
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


@dataclass
class DeviceList():
    """
    device list
    """
    name: str
    frequency: str
    arch: str
    bits: int
    count: int
    vendor_id: str
    device_list: List[Device] = field(default_factory=list)