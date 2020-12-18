from dataclasses import dataclass, field
from typing import List

@dataclass
class Process():

    username: str
    command: str
    full_command: str
    cpu_percent: float
    cpu_memory_usage: float
    pid: int
    gpu_memory_usage: float

