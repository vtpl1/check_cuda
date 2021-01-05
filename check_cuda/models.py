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
    name: str
    compute_capability_major: int
    compute_capability_minor: int
    cores: int
    concurrent_threads: int
    gpu_clock_mhz: int
    memory_clock_mhz: int
    total_memory_mib: int
    free_memory_mib: int


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
class NnModel(DataClassJsonMixin):
    """
    docstring
    """
    purpose: int
    width: int
    height: int


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NnModelMaxChannel(DataClassJsonMixin):
    """
    docstring
    """
    key: NnModel
    max_channel: int
    max_memory: int = 0
    max_fps: float = 0.0


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NnModelMaxChannelList(DataClassJsonMixin):
    models: List[NnModelMaxChannel] = field(default_factory=list)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(unsafe_hash=True)
class ChannelAndNnModel(DataClassJsonMixin):
    """
    docstring
    """
    channel_id: int
    model_id: NnModel


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


if __name__ == "__main__":
    l_m = NnModelMaxChannelList()
    l_m.models.append(NnModelMaxChannel(key=NnModel(75, 416, 416), max_channel=2))
    l_m.models.append(NnModelMaxChannel(key=NnModel(76, 416, 416), max_channel=3))

    # c = CpuStatus()
    # g = [GpuStatus(index=0)]
    # p = [ProcessStatus()]
    # s = SystemStatus(c, g, p)
    # x = s.to_dict()
    with open('data.yml', 'w') as outfile:
        yaml.dump(l_m.to_dict(), outfile)

    with open('data.yml', 'r') as infile:
        loaded = yaml.safe_load(infile)

        result1 = NnModelMaxChannelList.from_dict(loaded)
        print("MONOTOSH: ", type(result1))
        assert(isinstance(result1, NnModelMaxChannelList))
        pprint(result1)
