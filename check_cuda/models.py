from typing import List, Optional

from pydantic import BaseModel


class CpuInfo(BaseModel):
    """
    docstring
    """

    name: str = ""
    frequency: str = ""
    arch: str = ""
    bits: int = 64
    count: int = 0
    vendor_id: str = ""


class GpuInfo(BaseModel):
    """
    Nvidia device data
    """

    gpu_id: int
    name: Optional[str] = None
    total_memory_mib: Optional[int] = None
    free_memory_mib: Optional[int] = None

    uuid: Optional[str] = None
    compute_capability_major: Optional[int] = None
    compute_capability_minor: Optional[int] = None
    cores: Optional[int] = None
    concurrent_threads: Optional[int] = None
    gpu_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None


class SystemInfo(BaseModel):
    """
    docstring
    """

    host_name: str
    os: str
    cpu: CpuInfo
    gpus: List[GpuInfo]


class CpuStatus(BaseModel):
    """
    docstring
    """

    cpu_percent: float = 0.0
    cpu_memory_usage_percent: float = 0.0


class GpuStatus(BaseModel):
    """
    docstring
    """

    index: int
    uuid: Optional[str] = None
    name: Optional[str] = None
    temperature: Optional[float] = None
    fan_speed: Optional[float] = None
    utilization_gpu: Optional[float] = None
    utilization_enc: Optional[float] = None
    utilization_dec: Optional[float] = None
    power_draw: Optional[int] = None
    enforced_power_limit: Optional[int] = None
    memory_used: Optional[int] = None
    memory_total: Optional[int] = None


class ProcessStatus(BaseModel):
    """
    docstring
    """

    pid: int = 0
    command: Optional[str] = None
    full_command: Optional[str] = None
    cpu_percent: Optional[float] = None
    username: Optional[str] = None
    cpu_memory_usage_mib: Optional[int] = None
    gpu_memory_usage_mib: Optional[int] = None
    gpu_id: Optional[int] = None


class SystemStatus(BaseModel):
    """
    docstring
    """

    cpu: CpuStatus
    gpus: List[GpuStatus]
    processes: List[ProcessStatus]
