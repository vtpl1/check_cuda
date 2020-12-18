from check_cuda.data_models.nvidia_device import NvidiaDevice
from dataclasses import dataclass, field
from typing import List
from .intel_amd_device import IntelAmdDevice

@dataclass
class SystemDevice():
    """
    device list
    """
    cpus: List[IntelAmdDevice] = field(default_factory=list)
    gpus: List[NvidiaDevice] = field(default_factory=list)
