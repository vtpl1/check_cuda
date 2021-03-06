from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml
from dataclasses_json import DataClassJsonMixin, LetterCase, dataclass_json
from pprint import pprint


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CpuInfo(DataClassJsonMixin):
    """
    docstring
    """
    name: str = ""
    frequency: str = ""
    arch: str = ""
    bits: int = 64
    count: int = 0
    vendor_id: str = ""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class GpuInfo(DataClassJsonMixin):
    '''
    Nvidia device data
    '''
    gpu_id: int
    name: Optional[str] = None
    total_memory_mib: Optional[int] = None
    free_memory_mib: Optional[int] = None

    uuid: Optional[str] = None
    compute_capability_major: Optional[int] = None
    compute_capability_minor: Optional[int] = None
    cores: Optional[int] = None
    concurrent_threads: Optional[int] = None
    gpu_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class SystemInfo(DataClassJsonMixin):
    """
    docstring
    """
    host_name: str
    os: str
    cpu: CpuInfo
    gpus: List[GpuInfo] = field(default_factory=list)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CpuStatus(DataClassJsonMixin):
    """
    docstring
    """
    cpu_percent: float = 0.0
    cpu_memory_usage_percent: float = 0.0


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class GpuStatus(DataClassJsonMixin):
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


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ProcessStatus(DataClassJsonMixin):
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


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class SystemStatus(DataClassJsonMixin):
    """
    docstring
    """
    cpu: CpuStatus
    gpus: List[GpuStatus] = field(default_factory=list)
    processes: List[ProcessStatus] = field(default_factory=list)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(unsafe_hash=True)
class NnModelInfo(DataClassJsonMixin):
    """
    docstring
    """
    purpose: int
    width: int
    height: int
    max_fps: int
    memory: int

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NnModelStatus(DataClassJsonMixin):
    """
    docstring
    """
    key: NnModelInfo
    assigned_group_fps: int
    number_of_assigned_channels: int
    gpu_id: int
    channel_list: List[int] = field(default_factory=list)
    assigned_group_id_list: List[int] = field(default_factory=list)
     
    # max_channel: int
    # max_memory: int = 0
    # max_fps: float = 0.0

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NnModelMaxChannelInfo(DataClassJsonMixin):
    """
    docstring
    """
    key: NnModelInfo
    max_channel: int
    max_memory: int = 0
    max_fps: float = 0.0


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NnModelMaxChannelInfoList(DataClassJsonMixin):
    models: List[NnModelMaxChannelInfo] = field(default_factory=list)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(unsafe_hash=True)
class ChannelAndNnModel(DataClassJsonMixin):
    """
    docstring
    """
    channel_id: int
    model_id: NnModelInfo


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(unsafe_hash=True)
class ModelCount(DataClassJsonMixin):
    """
    docstring
    """
    gpu_id: int
    count: int = 1
    fps_consumed: float = 0


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelPerGpu(DataClassJsonMixin):
    """
    docstring
    """
