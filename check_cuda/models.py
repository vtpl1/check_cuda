from dataclasses import dataclass, field
from typing import List

@dataclass
class CpuInfo:
    """
    docstring
    """
    name: str = ""
    frequency: str = ""
    arch: str = ""
    bits: int = 64
    count: int = 0
    vendor_id: str = ""


@dataclass
class GpuInfo:
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


@dataclass
class SystemInfo:
    """
    docstring
    """
    host_name: str
    os: str
    cpu: CpuInfo
    gpus: List[GpuInfo] = field(default_factory=list)


@dataclass
class CpuStatus:
    """
    docstring
    """
    cpu_percent: float = 0.0
    cpu_memory_usage_percent: float = 0.0

@dataclass
class GpuStatus:
    """
    docstring
    """
    index: int
    uuid: str = None
    name: str = None
    temperature: float = None
    fan_speed: float = None
    utilization_gpu: float = None
    utilization_enc: float = None
    utilization_dec: float = None
    power_draw: int = None
    enforced_power_limit: int = None
    memory_used: int = None
    memory_total: int = None

@dataclass
class ProcessStatus:
    """
    docstring
    """
    pid: int = 0
    command: str = None
    full_command: str = None
    cpu_percent: float = 0.0
    username: str = None
    cpu_memory_usage_mib: int = None
    gpu_memory_usage_mib: int = None
    gpu_id: int = None


@dataclass
class SystemStatus:
    """
    docstring
    """
    cpu: CpuStatus
    gpus: List[GpuStatus] = field(default_factory=list)
    processes: List[ProcessStatus] = field(default_factory=list)
