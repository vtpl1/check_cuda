from dataclasses import dataclass, field
from typing import List


@dataclass
class CudaDevice():
    """
    device data
    """
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
    device_list: List[CudaDevice] = field(default_factory=list)