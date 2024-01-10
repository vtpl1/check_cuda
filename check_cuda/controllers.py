import logging
import os
import platform
from threading import Lock
from typing import List, Optional

import psutil
import pynvml as N
from cpuinfo import get_cpu_info

from .models import (
    CpuInfo,
    CpuStatus,
    GpuInfo,
    GpuStatus,
    ProcessStatus,
    SystemInfo,
    SystemStatus,
)

MB = 1024 * 1024

LOGGER = logging.getLogger(__name__)


def _extract_process_info(ps_process: psutil.Process) -> ProcessStatus:
    process = ProcessStatus()
    try:
        process.username = ps_process.username()
    except psutil.AccessDenied:
        pass

    # cmdline returns full path;,        # as in `ps -o comm`, get short cmdnames.
    _cmdline = None
    try:
        _cmdline = ps_process.cmdline()[0]
    except psutil.AccessDenied:
        pass

    if not _cmdline:
        # sometimes, zombie or unknown (e.g. [kworker/8:2H])
        process.command = "?"
        process.full_command = "?"
    else:
        process.command = os.path.basename(_cmdline)
        process.full_command = _cmdline
    try:
        cpu_percent = ps_process.cpu_percent(interval=0.1)
        process.cpu_percent = round(cpu_percent / psutil.cpu_count(), 1)
        process.cpu_memory_usage_mib = round(
            (ps_process.memory_percent() / 100.0) * psutil.virtual_memory().total // MB
        )
    except psutil.AccessDenied:
        pass
    process.pid = ps_process.pid
    return process


def get_process_status_by_pid(pid) -> ProcessStatus:
    ps_process = psutil.Process(pid=pid)
    process = _extract_process_info(ps_process)
    return process


def get_process_status_by_name(name="python") -> List[ProcessStatus]:
    process_list = []
    for ps_process in psutil.process_iter():
        name_, exe, cmdline = "", "", []
        try:
            name_ = ps_process.name()
            cmdline = ps_process.cmdline()
            exe = ps_process.exe()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except psutil.NoSuchProcess:
            continue
        if len(cmdline):
            if name == name_ or cmdline[0] == name or os.path.basename(exe) == name:
                process_list.append(_extract_process_info(ps_process))
    return process_list


class GpuInfoFromNvml:
    __singleton_lock = Lock()
    __singleton_instance = None

    def __init__(self):
        self.__is_nvml_loaded = False
        self.__gpu_processes: List[ProcessStatus] = []
        print("Starting NVML")
        try:
            N.nvmlInit()
            self.__is_nvml_loaded = True
        except Exception as e:
            print(e)

    def __del__(self):
        print("Shutting down NVML")
        if self.__is_nvml_loaded:
            if N:
                N.nvmlShutdown()

    @classmethod
    def instance(cls):
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()
        return cls.__singleton_instance

    def _decode(self, b):
        if isinstance(b, bytes):
            return b.decode("utf-8")  # for python3, to unicode
        return b

    def get_process_status_running_on_gpus(self) -> List[ProcessStatus]:
        ret = []
        for gpu_process in self.__gpu_processes:
            ret.append(get_process_status_by_pid(gpu_process.pid))
        return ret

    def get_gpu_status_by_gpu_id(self, index) -> Optional[GpuStatus]:
        gpu_status = None
        if self.__is_nvml_loaded:
            gpu_status = GpuStatus(index=index)
            """Get one GPU information specified by nvml handle"""
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_status.name = self._decode(N.nvmlDeviceGetName(handle))
            gpu_status.uuid = self._decode(N.nvmlDeviceGetUUID(handle))
            try:
                gpu_status.temperature = N.nvmlDeviceGetTemperature(
                    handle, N.NVML_TEMPERATURE_GPU
                )
            except N.NVMLError:
                gpu_status.temperature = None  # Not supported
            try:
                gpu_status.fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            except N.NVMLError:
                gpu_status.fan_speed = None  # Not supported
            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
                gpu_status.memory_used = memory.used // MB
                gpu_status.memory_total = memory.total // MB
            except N.NVMLError:
                gpu_status.memory_used = None  # Not supported
                gpu_status.memory_total = None

            try:
                utilization = N.nvmlDeviceGetUtilizationRates(handle)
                if utilization:
                    gpu_status.utilization_gpu = utilization.gpu
                else:
                    gpu_status.utilization_gpu = None
            except N.NVMLError:
                gpu_status.utilization_gpu = None  # Not supported

            try:
                utilization_enc = N.nvmlDeviceGetEncoderUtilization(handle)
                if utilization_enc:
                    gpu_status.utilization_enc = utilization_enc[0]
                else:
                    gpu_status.utilization_enc = None
            except N.NVMLError:
                gpu_status.utilization_enc = None  # Not supported

            try:
                utilization_dec = N.nvmlDeviceGetDecoderUtilization(handle)
                if utilization_dec:
                    gpu_status.utilization_dec = utilization_dec[0]
                else:
                    gpu_status.utilization_dec = None
            except N.NVMLError:
                gpu_status.utilization_dec = None  # Not supported

            try:
                nv_comp_processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError:
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError:
                nv_graphics_processes = None  # Not supported
            if nv_comp_processes is None and nv_graphics_processes is None:
                pass
            else:
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                # A single process might run in both of graphics and compute mode,
                # However we will display the process only once
                seen_pids = set()
                for nv_process in nv_comp_processes + nv_graphics_processes:
                    if nv_process.pid in seen_pids:
                        continue
                    seen_pids.add(nv_process.pid)
                    try:
                        process = get_process_status_by_pid(nv_process.pid)
                        # Bytes to MBytes
                        # if drivers are not TTC this will be None.
                        usedmem = (
                            nv_process.usedGpuMemory // MB
                            if nv_process.usedGpuMemory
                            else None
                        )
                        process.gpu_memory_usage_mib = usedmem
                        process.gpu_id = index
                        self.__gpu_processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass
        return gpu_status

    def get_gpu_info_by_gpu_id(self, index) -> Optional[GpuInfo]:
        gpu_info = None
        if self.__is_nvml_loaded:
            gpu_info = GpuInfo(gpu_id=index)
            """Get one GPU information specified by nvml handle"""
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_info.name = self._decode(N.nvmlDeviceGetName(handle))
            gpu_info.uuid = self._decode(N.nvmlDeviceGetUUID(handle))

            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
                gpu_info.free_memory_mib = (memory.total - memory.used) // MB
                gpu_info.total_memory_mib = memory.total // MB
            except N.NVMLError:
                gpu_info.free_memory_mib = None  # Not supported
                gpu_info.total_memory_mib = None

        return gpu_info

    def get_gpu_info(self) -> List[GpuInfo]:
        gpu_list = []
        if self.__is_nvml_loaded:
            device_count = N.nvmlDeviceGetCount()
            for index in range(device_count):
                gpu_status = self.get_gpu_info_by_gpu_id(index)
                if gpu_status:
                    gpu_list.append(gpu_status)
        return gpu_list

    def get_gpu_status(self) -> List[GpuStatus]:
        gpu_list = []
        if self.__is_nvml_loaded:
            device_count = N.nvmlDeviceGetCount()
            self.__gpu_processes.clear()
            for index in range(device_count):
                gpu_status = self.get_gpu_status_by_gpu_id(index)
                if gpu_status:
                    gpu_list.append(gpu_status)
        return gpu_list


def get_system_info() -> SystemInfo:
    return SystemInfo(
        host_name=platform.uname().node,
        os=platform.platform(),
        cpu=get_cpu(),
        gpus=get_gpu_info(),
    )


def get_cpu() -> CpuInfo:
    cpu = CpuInfo()
    try:
        cpu_info = get_cpu_info()
        cpu.name = cpu_info["brand_raw"]
        cpu.frequency = cpu_info["hz_advertised_friendly"]
        cpu.arch = cpu_info["arch"]
        cpu.bits = cpu_info["bits"]
        cpu.count = cpu_info["count"]
        cpu.vendor_id = cpu_info["vendor_id_raw"]
    except AttributeError as e:
        LOGGER.fatal(e)
    return cpu


def get_gpu_info() -> List[GpuInfo]:
    return GpuInfoFromNvml.instance().get_gpu_info()


def get_cpu_status() -> CpuStatus:
    return CpuStatus(
        cpu_percent=psutil.cpu_percent(),
        cpu_memory_usage_percent=psutil.virtual_memory().percent,
    )


def get_gpu_status() -> List[GpuStatus]:
    return GpuInfoFromNvml.instance().get_gpu_status()


def get_process_status_running_on_gpus() -> List[ProcessStatus]:
    return GpuInfoFromNvml.instance().get_process_status_running_on_gpus()


def get_process_status() -> List[ProcessStatus]:
    ret = get_process_status_running_on_gpus()
    if not len(ret):
        ret = get_process_status_by_name()
    return ret


def get_system_status() -> SystemStatus:
    return SystemStatus(
        cpu=get_cpu_status(), gpus=get_gpu_status(), processes=get_process_status()
    )
