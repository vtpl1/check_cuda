from check_cuda.data_models.nvidia_device import NvidiaDevice
from check_cuda.data_models.intel_amd_device import IntelAmdDevice
from cpuinfo import get_cpu_info
import logging
from .data_models.system_device import SystemDevice
from .check_cuda import get_nvidia_device_info
from multiprocessing import Process, freeze_support

LOGGER = logging.getLogger(__name__)


def get_hardware_info() -> SystemDevice:
    cpu_name = ""
    cpu_frequency = ""
    cpu_arch = ""
    cpu_bits = 64
    cpu_count = 0
    cpu_vendor_id = ""
    try:
        cpu_info = get_cpu_info()
        cpu_name = cpu_info["brand_raw"]
        cpu_frequency = cpu_info["hz_advertised_friendly"]
        cpu_arch = cpu_info["arch"]
        cpu_bits = cpu_info["bits"]
        cpu_count = cpu_info["count"]
        cpu_vendor_id = cpu_info["vendor_id_raw"]

    except AttributeError as e:
        LOGGER.fatal(e)
    except Exception as e1:
        LOGGER.fatal(e1)

    cpus = []
    cpus.append(
        IntelAmdDevice(
            cpu_name, cpu_frequency, cpu_arch, cpu_bits, cpu_count, cpu_vendor_id
        )
    )
    gpus = get_nvidia_device_info()
    ret = SystemDevice(cpus=cpus, gpus=gpus)
    return ret