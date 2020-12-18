from dataclasses import dataclass, field
from typing import List

@dataclass
class IntelAmdDevice(object):
    name: str
    frequency: str
    arch: str
    bits: int
    count: int
    vendor_id: str